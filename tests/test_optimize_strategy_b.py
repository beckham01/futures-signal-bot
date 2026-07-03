import pandas as pd

from backtest.optimize_strategy_b import (
    StrategyBMetrics,
    build_candidates_for_symbol,
    candidate_accepted,
    evaluate_candidates_fast,
    metrics_from_results,
    parameter_grid,
    score_candidate,
    walk_forward_bounds,
)
from backtest.simulator import TradeResult
from backtest.strategy import SignalEvent


def strategy_b_test_cfg(**overrides):
    base = {
        "timeframe": "15",
        "ema_slow": 55,
        "rsi_period": 14,
        "atr_period": 14,
        "volume_sma_period": 20,
        "volume_spike_threshold": 1.3,
        "volume_strong_threshold": 1.6,
        "swing_lookback": 8,
        "ema_slope_lookback": 5,
        "candle_body_min_pct": 0.5,
        "candle_body_strong_pct": 0.65,
        "breakout_margin_atr": 0.3,
        "tp1_risk_multiplier": 1.5,
        "tp2_risk_multiplier": 2.5,
        "sl_atr_buffer": 0.5,
        "min_risk_reward": 2.0,
        "min_confidence": 50,
        "atr_max_pct": 0.04,
        "atr_min_pct": 0.003,
        "cooldown_hours": 4,
    }
    base.update(overrides)
    return base


def frame():
    periods = 14
    idx = 10
    timestamps = pd.date_range("2026-01-01T00:00:00Z", periods=periods, freq="15min")
    close = [100.0 + i * 0.04 for i in range(periods)]
    open_ = [value - 0.1 for value in close]
    high = [value + 0.2 for value in close]
    low = [value - 0.2 for value in close]
    close[idx] = max(high[idx - 8 : idx]) + 0.6
    open_[idx] = close[idx] - 0.8
    high[idx] = close[idx] + 0.1
    low[idx] = close[idx] - 1.0
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": [100.0] * idx + [180.0] + [100.0] * (periods - idx - 1),
            "ema55": [99.0 + i * 0.02 for i in range(periods)],
            "ema55_slope": [0.02] * periods,
            "rsi14": [52.0] * idx + [62.0] + [52.0] * (periods - idx - 1),
            "atr14": [1.0] * periods,
            "volume_sma20": [100.0] * periods,
            "atr_expanding": [True] * periods,
        }
    )


def make_result(symbol="BTCUSDT", outcome="TP2_HIT", pnl_r=2.0, timestamp="2026-01-01T00:00:00Z"):
    signal = SignalEvent(
        symbol=symbol,
        direction="LONG",
        timestamp=pd.Timestamp(timestamp),
        entry=100.0,
        stop_loss=98.0,
        tp1=103.0,
        tp2=105.0,
        risk_reward=2.5,
        confidence=80,
        confidence_label="STRONG",
        atr_15m=1.0,
        reasons=[],
        strategy_name="strategy_b_daily_momentum",
    )
    return TradeResult(signal, outcome, pnl_r, 4, 105.0, signal.timestamp + pd.Timedelta(hours=1))


def test_strategy_b_metrics_and_acceptance():
    results = [make_result("BTCUSDT", "TP2_HIT", 2.0), make_result("ETHUSDT", "STOP_HIT", -1.0)]
    metrics = metrics_from_results(results, days=7)
    assert metrics.trades == 2
    assert metrics.trades_per_week == 2
    assert metrics.win_rate == 50

    accepted = StrategyBMetrics(30, 5.0, 45.0, 4.0, 0.13, 4, 0.4)
    rejected = StrategyBMetrics(30, 12.0, 45.0, 4.0, 0.13, 4, 0.4)
    assert candidate_accepted(accepted) is True
    assert candidate_accepted(rejected) is False


def test_strategy_b_score_prefers_positive_validation():
    train = StrategyBMetrics(30, 5.0, 45.0, 4.0, 0.13, 4, 0.4)
    better = StrategyBMetrics(20, 5.0, 50.0, 5.0, 0.25, 3, 0.4)
    worse = StrategyBMetrics(20, 9.0, 40.0, -1.0, -0.05, 8, 0.7)
    assert score_candidate(train, better) > score_candidate(train, worse)


def test_strategy_b_walk_forward_bounds():
    timestamps = pd.date_range("2026-01-01", periods=11, freq="15min", tz="UTC")
    data = {
        "BTCUSDT": {"15": pd.DataFrame({"timestamp": timestamps})},
        "ETHUSDT": {"15": pd.DataFrame({"timestamp": timestamps[1:]})},
    }
    start, split, end = walk_forward_bounds(data, ["BTCUSDT", "ETHUSDT"], "15")
    assert start == timestamps[1]
    assert end == timestamps[-1]
    assert start < split < end


def test_strategy_b_strict_grid_is_more_selective_than_default():
    strict = parameter_grid("strict")[0]
    default = parameter_grid("default")[0]
    assert strict["swing_lookback"] > default["swing_lookback"]
    assert strict["cooldown_hours"] > default["cooldown_hours"]


def test_fast_candidate_evaluation_returns_strategy_b_signal():
    data = frame()
    config = strategy_b_test_cfg()
    candidates = {"BTCUSDT": build_candidates_for_symbol("BTCUSDT", data, config)}
    signals = evaluate_candidates_fast(candidates, ["BTCUSDT"], config)
    assert len(signals) == 1
    assert signals[0].strategy_name == "strategy_b_daily_momentum"
