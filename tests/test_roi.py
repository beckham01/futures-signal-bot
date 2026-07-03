import pandas as pd
import pytest

from backtest.roi import (
    generate_roi_report,
    required_price_move_pct,
    roi_label_for_tiers,
    roi_metrics,
    roi_target_price,
    simulate_roi_trade,
)
from backtest.strategy import SignalEvent


def signal(direction="LONG"):
    return SignalEvent(
        symbol="BTCUSDT",
        direction=direction,
        timestamp=pd.Timestamp("2026-01-01T00:00:00Z"),
        entry=100.0,
        stop_loss=90.0 if direction == "LONG" else 110.0,
        tp1=110.0,
        tp2=120.0,
        risk_reward=2.0,
        confidence=90,
        confidence_label="STRONG",
        atr_15m=1.0,
        reasons=[],
    )


def candles(rows):
    return pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-01-01T00:15:00Z") + pd.Timedelta(minutes=15 * index),
                "open": row[0],
                "high": row[1],
                "low": row[2],
                "close": row[3],
                "volume": 1.0,
            }
            for index, row in enumerate(rows)
        ]
    )


def test_required_price_move_at_5x():
    assert required_price_move_pct(100, 5) == 20
    assert roi_target_price(100, "LONG", 100, 5) == 120
    assert roi_target_price(100, "SHORT", 100, 5) == 80


def test_long_roi_tiers_detected():
    result = simulate_roi_trade(signal(), candles([(100, 141, 99, 140)]), leverage=5)
    assert result.reached_tiers[100] is True
    assert result.reached_tiers[200] is True
    assert result.max_favorable_roi_pct == 205.0


def test_short_roi_tiers_detected():
    result = simulate_roi_trade(signal("SHORT"), candles([(100, 101, 60, 61)]), leverage=5)
    assert result.reached_tiers[100] is True
    assert result.reached_tiers[200] is True
    assert result.max_favorable_roi_pct == 200.0


def test_roi_label_for_highest_tier():
    assert roi_label_for_tiers({100: True, 200: True, 300: False, 400: False, 500: False}) == "ROI200_STRETCH"
    assert roi_label_for_tiers({100: True, 200: True, 300: True, 400: True, 500: True}) == "ROI500_EXTREME"


def test_impossible_short_target_rejected():
    with pytest.raises(ValueError):
        roi_target_price(100, "SHORT", 500, 5)


def test_roi_report_includes_hit_rates(tmp_path):
    result = simulate_roi_trade(signal(), candles([(100, 121, 99, 120)]), leverage=5)
    metrics = roi_metrics([result])
    assert metrics.roi100_hit_rate == 100
    output = tmp_path / "roi.txt"
    report = generate_roi_report([result], "test", 5, output)
    assert "+100% ROI hit rate" in report
    assert "MFE" not in report
    assert output.exists()
