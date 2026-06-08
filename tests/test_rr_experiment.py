import pandas as pd

from backtest.rr_experiment import adjusted_signal, apply_scenario, is_structure_extension_candidate
from backtest.strategy import SignalEvent


def make_signal(confidence=100, conditions=None):
    return SignalEvent(
        symbol="BTCUSDT",
        direction="LONG",
        timestamp=pd.Timestamp("2026-01-01T00:00:00Z"),
        entry=100.0,
        stop_loss=98.0,
        tp1=104.0,
        tp2=106.0,
        risk_reward=3.0,
        confidence=confidence,
        confidence_label="STRONG",
        atr_15m=1.0,
        reasons=[],
        confidence_conditions=conditions
        or {
            "trend_rsi_strong": True,
            "atr_expanding": True,
            "ema55_slope": True,
            "strong_volume": True,
        },
    )


def test_adjusted_signal_moves_tp2_to_target_rr():
    signal = adjusted_signal(make_signal(), 5.0)

    assert signal.tp2 == 110.0
    assert signal.risk_reward == 5.0


def test_structure_extension_candidate_requires_quality_conditions():
    assert is_structure_extension_candidate(make_signal())
    assert not is_structure_extension_candidate(make_signal(confidence=80))
    assert not is_structure_extension_candidate(make_signal(conditions={"trend_rsi_strong": True}))


def test_apply_scenario_keeps_base_when_predicate_false():
    signals = apply_scenario([make_signal()], 3.0, 10.0, lambda signal: False)

    assert signals[0].risk_reward == 3.0
    assert signals[0].tp2 == 106.0
