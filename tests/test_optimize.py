import pandas as pd

from backtest.optimize import (
    Metrics,
    build_symbol_sets,
    candidate_accepted,
    evaluate_prepared_signals,
    evaluate_signal_candidates,
    metrics_from_results,
    prepare_merged_frame,
    prepare_optimizer_data,
    score_candidate,
    simulate_all_fast,
    walk_forward_bounds,
)
from backtest.simulator import TradeResult, simulate_all
from backtest.strategy import SignalEvent, evaluate_signals
from tests.test_strategy import prepared_frames


def make_result(symbol="BTCUSDT", outcome="TP2_HIT", pnl_r=2.17, timestamp="2026-01-01T00:00:00Z"):
    signal = SignalEvent(
        symbol=symbol,
        direction="LONG",
        timestamp=pd.Timestamp(timestamp),
        entry=100.0,
        stop_loss=98.5,
        tp1=102.0,
        tp2=104.5,
        risk_reward=3.0,
        confidence=85,
        confidence_label="STRONG",
        atr_15m=1.0,
        reasons=[],
    )
    return TradeResult(signal, outcome, pnl_r, 4, 104.5, pd.Timestamp(timestamp) + pd.Timedelta(hours=1))


def test_build_symbol_sets_quarantines_weak_symbols():
    sets = build_symbol_sets(["BTCUSDT", "ADAUSDT", "DOGEUSDT", "ETHUSDT"])
    assert sets["all_symbols"] == ["BTCUSDT", "ADAUSDT", "DOGEUSDT", "ETHUSDT"]
    assert sets["quarantine_weak"] == ["BTCUSDT", "ETHUSDT"]


def test_walk_forward_bounds_uses_shared_range():
    timestamps = pd.date_range("2026-01-01", periods=11, freq="1h", tz="UTC")
    data = {
        "BTCUSDT": {"15": pd.DataFrame({"timestamp": timestamps})},
        "ETHUSDT": {"15": pd.DataFrame({"timestamp": timestamps[1:]})},
    }
    start, split, end = walk_forward_bounds(data, ["BTCUSDT", "ETHUSDT"], "15")
    assert start == timestamps[1]
    assert end == timestamps[-1]
    assert start < split < end


def test_metrics_and_acceptance():
    results = [
        make_result("BTCUSDT", "TP2_HIT", 2.17, "2026-01-01T00:00:00Z"),
        make_result("ETHUSDT", "TP1_ONLY", 0.665, "2026-01-01T01:00:00Z"),
        make_result("SOLUSDT", "STOP_HIT", -1.0, "2026-01-01T02:00:00Z"),
    ]
    metrics = metrics_from_results(results)
    assert round(metrics.win_rate, 1) == 66.7
    assert metrics.max_consecutive_losses == 1

    accepted = Metrics(30, 55.0, 5.0, 0.16, 5, 0.4)
    rejected = Metrics(30, 45.0, 5.0, 0.16, 5, 0.4)
    assert candidate_accepted(accepted) is True
    assert candidate_accepted(rejected) is False


def test_score_candidate_prefers_better_validation():
    train = Metrics(40, 52.0, 4.0, 0.1, 5, 0.4)
    better = Metrics(35, 60.0, 8.0, 0.22, 4, 0.4)
    worse = Metrics(35, 50.0, 2.0, 0.05, 8, 0.7)
    assert score_candidate(train, better) > score_candidate(train, worse)


def test_prepared_evaluation_matches_normal_strategy():
    df_15m, df_1h = prepared_frames("LONG")
    cfg = {
        "volume_spike_threshold": 1.3,
        "volume_strong_threshold": 1.5,
        "pullback_atr_tolerance": 0.5,
        "rsi_long_threshold": 45,
        "rsi_short_threshold": 55,
        "tp1_atr_multiplier": 2.0,
        "tp2_atr_multiplier": 4.5,
        "sl_atr_multiplier": 1.5,
        "min_risk_reward": 3.0,
        "min_confidence": 55,
        "atr_max_pct": 0.04,
        "atr_min_pct": 0.003,
        "cooldown_hours": 4,
    }
    normal = evaluate_signals("BTCUSDT", df_15m.copy(), df_1h.copy(), cooldown_hours=4, cfg=cfg)
    merged = prepare_merged_frame(df_15m.copy(), df_1h.copy())
    prepared = evaluate_prepared_signals(
        "BTCUSDT",
        merged,
        cfg,
        df_15m["timestamp"].min(),
        df_15m["timestamp"].max(),
    )
    symbol_data = prepare_optimizer_data(
        {"BTCUSDT": {"15": df_15m.copy(), "60": df_1h.copy()}},
        ["BTCUSDT"],
        "15",
        "60",
    )["BTCUSDT"]
    candidate_signals = evaluate_signal_candidates(
        symbol_data.candidates,
        cfg,
        df_15m["timestamp"].min(),
        df_15m["timestamp"].max(),
    )
    assert [(s.direction, s.timestamp, s.entry, s.confidence) for s in prepared] == [
        (s.direction, s.timestamp, s.entry, s.confidence) for s in normal
    ]
    assert [(s.direction, s.timestamp, s.entry, s.confidence) for s in candidate_signals] == [
        (s.direction, s.timestamp, s.entry, s.confidence) for s in normal
    ]


def test_fast_simulation_matches_normal_simulation():
    timestamps = pd.date_range("2026-01-01T00:00:00Z", periods=4, freq="15min")
    candles = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100.0, 100.0, 102.0, 104.0],
            "high": [100.5, 102.5, 105.0, 105.0],
            "low": [99.5, 99.5, 101.0, 103.0],
            "close": [100.0, 102.0, 104.5, 104.0],
            "volume": [1.0] * 4,
        }
    )
    signal = make_result(timestamp="2026-01-01T00:00:00Z").signal
    data = {"BTCUSDT": {"15": candles}}
    prepared = prepare_optimizer_data({"BTCUSDT": {"15": candles, "60": candles}}, ["BTCUSDT"], "15", "60")

    normal = simulate_all([signal], data)
    fast = simulate_all_fast([signal], prepared)

    assert [(r.outcome, r.pnl_r, r.bars_held) for r in fast] == [
        (r.outcome, r.pnl_r, r.bars_held) for r in normal
    ]
