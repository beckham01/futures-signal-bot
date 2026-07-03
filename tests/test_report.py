import pandas as pd

from backtest.report import (
    confidence_bucket,
    generate_comparison_report,
    generate_report,
    generate_report_b,
    generate_report_c,
    summarize_by_confidence_condition,
)
from backtest.simulator import TradeResult
from backtest.strategy import SignalEvent


def make_result(score=85, outcome="TP2_HIT", pnl_r=2.17):
    signal = SignalEvent(
        symbol="BTCUSDT",
        direction="LONG",
        timestamp=pd.Timestamp("2026-01-01T00:00:00Z"),
        entry=100.0,
        stop_loss=98.5,
        tp1=102.0,
        tp2=104.5,
        risk_reward=3.0,
        confidence=score,
        confidence_label="STRONG",
        atr_15m=1.0,
        reasons=[],
        confidence_conditions={"strong_volume": True, "atr_expanding": False},
    )
    return TradeResult(signal, outcome, pnl_r, 4, 104.5, pd.Timestamp("2026-01-01T01:00:00Z"))


def test_confidence_bucket():
    assert confidence_bucket(100) == "95-100"
    assert confidence_bucket(87) == "85-94"
    assert confidence_bucket(60) == "55-64"


def test_summarize_by_confidence_condition():
    summary = summarize_by_confidence_condition([make_result()])
    assert summary["strong_volume"][0] == 1
    assert "atr_expanding" not in summary


def test_generate_report_includes_diagnostics(tmp_path):
    output = tmp_path / "report.txt"
    report = generate_report(
        [make_result()],
        ["BTCUSDT"],
        "2026-01-01",
        "2026-01-02",
        output_path=output,
    )
    assert "--- BY DIRECTION ---" in report
    assert "--- BY CONFIDENCE SCORE BUCKET ---" in report
    assert "--- BY CONFIDENCE CONDITION TRUE ---" in report
    assert output.exists()


def test_generate_report_b_includes_acceptance(tmp_path):
    output = tmp_path / "strategy_b.txt"
    report = generate_report_b(
        [make_result() for _ in range(80)],
        "2026-01-01",
        "2026-07-01",
        ["BTCUSDT"],
        days=180,
        output_path=output,
    )
    assert "--- STRATEGY B ACCEPTANCE ---" in report
    assert "Verdict:" in report
    assert output.exists()


def test_generate_report_c_includes_acceptance(tmp_path):
    output = tmp_path / "strategy_c.txt"
    report = generate_report_c(
        [make_result(pnl_r=1.0) for _ in range(80)],
        "2026-01-01",
        "2026-07-01",
        ["BTCUSDT"],
        days=180,
        output_path=output,
    )
    assert "--- STRATEGY C ACCEPTANCE ---" in report
    assert "Verdict:" in report
    assert output.exists()


def test_generate_comparison_report_includes_verdict(tmp_path):
    output = tmp_path / "comparison.txt"
    results_a = [make_result() for _ in range(30)]
    results_b = [make_result() for _ in range(80)]
    report = generate_comparison_report(
        results_a,
        results_b,
        [],
        results_a + results_b,
        [("BTCUSDT", pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:30:00Z"))],
        "2026-01-01",
        "2026-07-01",
        days=180,
        output_path=output,
    )
    assert "STRATEGY COMPARISON REPORT" in report
    assert "Conflicts detected: 1" in report
    assert "Strategy C" in report
    assert "Combined improvement over A alone" in report
