import pandas as pd

from backtest.analyze_optimization import (
    analyze_results,
    candidate_config_from_row,
    classify_tier,
    generate_report,
    load_results,
    practical_score,
    write_candidate_config,
)


def row(**overrides):
    data = {
        "name": "core_liquid",
        "symbols": "BTCUSDT ETHUSDT SOLUSDT",
        "accepted": False,
        "score": 50.0,
        "volume_spike_threshold": 1.5,
        "pullback_atr_tolerance": 0.35,
        "cooldown_hours": 4,
        "atr_min_pct": 0.003,
        "atr_max_pct": 0.03,
        "min_confidence": 55,
        "train_trades": 25,
        "train_win_rate": 70.0,
        "train_total_r": 8.0,
        "train_avg_r": 0.32,
        "train_max_consecutive_losses": 2,
        "train_top_symbol_trade_share": 0.4,
        "validation_trades": 9,
        "validation_win_rate": 66.7,
        "validation_total_r": 7.0,
        "validation_avg_r": 0.78,
        "validation_max_consecutive_losses": 2,
        "validation_top_symbol_trade_share": 0.44,
        "source_file": "input.csv",
    }
    data.update(overrides)
    return pd.Series(data)


def test_classify_tiers():
    assert classify_tier(row(validation_trades=2, validation_win_rate=100.0)) == "TINY_SAMPLE_TRAP"
    assert classify_tier(row()) == "PRACTICAL_REVIEW"
    assert classify_tier(row(validation_trades=30, validation_win_rate=55.0, validation_top_symbol_trade_share=0.5)) == "ACCEPTED"
    assert classify_tier(row(validation_win_rate=45.0)) == "REJECTED"


def test_practical_score_prefers_better_candidate():
    better = row(validation_trades=12, validation_win_rate=70.0, validation_avg_r=0.8)
    worse = row(validation_trades=8, validation_win_rate=60.0, validation_avg_r=0.3)
    assert practical_score(better) > practical_score(worse)


def test_load_results_multi_file(tmp_path):
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    pd.DataFrame([row().to_dict()]).to_csv(first, index=False)
    pd.DataFrame([row(name="quarantine_weak").to_dict()]).to_csv(second, index=False)

    loaded = load_results([first, second])

    assert len(loaded) == 2
    assert set(loaded["source_file"]) == {"first.csv", "second.csv"}


def test_generate_report_mentions_tiny_sample_rejection():
    analyzed = analyze_results(
        pd.DataFrame(
            [
                row().to_dict(),
                row(validation_trades=2, validation_win_rate=100.0, score=120.0).to_dict(),
            ]
        )
    )

    report = generate_report(analyzed, ["sample.csv"])

    assert "Tiny-sample traps" in report
    assert "validation_trades < 8" in report
    assert "TOP PRACTICAL CANDIDATES" in report


def test_candidate_config_generation(tmp_path):
    base_config = {
        "watchlist": ["OLDUSDT"],
        "strategy": {
            "volume_spike_threshold": 1.3,
            "pullback_atr_tolerance": 0.5,
            "atr_min_pct": 0.003,
            "atr_max_pct": 0.04,
            "min_confidence": 55,
        },
        "bot": {"cooldown_hours": 4},
    }
    config = candidate_config_from_row(row(), base_config)
    assert config["watchlist"] == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    assert config["strategy"]["volume_spike_threshold"] == 1.5
    assert config["bot"]["cooldown_hours"] == 4
    assert "candidate_note" in config

    output = tmp_path / "candidate.yaml"
    analyzed = analyze_results(pd.DataFrame([row().to_dict()]))
    assert write_candidate_config(analyzed, output, base_config) is True
    assert output.exists()
