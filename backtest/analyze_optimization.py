"""Analyze optimizer CSV outputs and produce review-ready candidate reports."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from backtest.strategy import load_config

BASELINE_WIN_RATE = 49.0
BASELINE_MAX_LOSSES = 9
PARAM_COLUMNS = [
    "volume_spike_threshold",
    "pullback_atr_tolerance",
    "cooldown_hours",
    "atr_min_pct",
    "atr_max_pct",
    "min_confidence",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze optimizer CSV outputs.")
    parser.add_argument("inputs", nargs="+", help="Optimizer CSV files to analyze")
    parser.add_argument("--output", default="optimization_analysis.csv")
    parser.add_argument("--report", default="optimization_analysis_report.txt")
    parser.add_argument("--candidate-config", default="candidate_config.yaml")
    return parser.parse_args()


def load_results(paths: list[str | Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        source = Path(path)
        frame = pd.read_csv(source)
        frame["source_file"] = source.name
        frames.append(frame)
    if not frames:
        raise ValueError("At least one optimizer CSV is required")
    return pd.concat(frames, ignore_index=True)


def classify_tier(row: pd.Series) -> str:
    if int(row["validation_trades"]) < 8:
        return "TINY_SAMPLE_TRAP"
    accepted = (
        row["validation_trades"] >= 30
        and row["validation_win_rate"] > BASELINE_WIN_RATE
        and row["validation_total_r"] > 0
        and row["validation_avg_r"] > 0
        and row["validation_max_consecutive_losses"] <= BASELINE_MAX_LOSSES
        and row["validation_top_symbol_trade_share"] <= 0.60
    )
    if accepted:
        return "ACCEPTED"
    practical_review = (
        row["validation_trades"] >= 8
        and row["train_trades"] >= 20
        and row["validation_win_rate"] >= 60
        and row["validation_total_r"] > 0
        and row["validation_avg_r"] > 0.25
        and row["validation_max_consecutive_losses"] <= 3
        and row["validation_top_symbol_trade_share"] <= 0.50
    )
    if practical_review:
        return "PRACTICAL_REVIEW"
    return "REJECTED"


def practical_score(row: pd.Series) -> float:
    train_validation_gap = abs(float(row["train_win_rate"]) - float(row["validation_win_rate"]))
    trade_bonus = min(float(row["validation_trades"]), 30.0) * 0.7
    return (
        float(row["validation_win_rate"])
        + float(row["validation_avg_r"]) * 25
        + trade_bonus
        - float(row["validation_max_consecutive_losses"]) * 2.5
        - max(float(row["validation_top_symbol_trade_share"]) - 0.45, 0) * 25
        - train_validation_gap * 0.35
    )


def analyze_results(df: pd.DataFrame) -> pd.DataFrame:
    analyzed = df.copy()
    analyzed["tier"] = analyzed.apply(classify_tier, axis=1)
    analyzed["train_validation_win_gap"] = (
        analyzed["train_win_rate"] - analyzed["validation_win_rate"]
    ).abs()
    analyzed["practical_score"] = analyzed.apply(practical_score, axis=1)
    return analyzed.sort_values(
        ["tier", "practical_score", "validation_trades"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def top_practical_candidates(analyzed: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    practical = analyzed[analyzed["tier"].isin(["ACCEPTED", "PRACTICAL_REVIEW"])]
    return practical.sort_values(
        ["practical_score", "validation_trades", "validation_win_rate"],
        ascending=[False, False, False],
    ).head(limit)


def parameter_stability_summary(analyzed: pd.DataFrame) -> list[str]:
    practical = analyzed[analyzed["tier"].isin(["ACCEPTED", "PRACTICAL_REVIEW"])]
    if practical.empty:
        return ["No accepted or practical-review candidates were found."]
    lines = []
    for column in PARAM_COLUMNS:
        counts = practical[column].value_counts().sort_index()
        rendered = ", ".join(f"{value}: {count}" for value, count in counts.items())
        lines.append(f"{column}: {rendered}")
    return lines


def generate_report(analyzed: pd.DataFrame, input_paths: list[str | Path]) -> str:
    tier_counts = analyzed["tier"].value_counts()
    top = top_practical_candidates(analyzed)
    tiny = analyzed[analyzed["tier"] == "TINY_SAMPLE_TRAP"].sort_values("score", ascending=False).head(5)
    lines = [
        "=== OPTIMIZATION ANALYSIS REPORT ===",
        f"Files analyzed: {', '.join(str(path) for path in input_paths)}",
        f"Total rows: {len(analyzed)}",
        f"Accepted: {int(tier_counts.get('ACCEPTED', 0))}",
        f"Practical review: {int(tier_counts.get('PRACTICAL_REVIEW', 0))}",
        f"Tiny-sample traps: {int(tier_counts.get('TINY_SAMPLE_TRAP', 0))}",
        f"Rejected: {int(tier_counts.get('REJECTED', 0))}",
        "",
        "--- TOP PRACTICAL CANDIDATES ---",
    ]
    if top.empty:
        lines.append("No candidates met the Practical Review filters.")
    else:
        for _, row in top.iterrows():
            lines.append(
                f"{row['tier']} | {row['name']} | score {row['practical_score']:.2f} | "
                f"val trades {int(row['validation_trades'])} | win {row['validation_win_rate']:.1f}% | "
                f"avg R {row['validation_avg_r']:.2f} | params "
                f"vol={row['volume_spike_threshold']} pullback={row['pullback_atr_tolerance']} "
                f"cooldown={int(row['cooldown_hours'])} conf={int(row['min_confidence'])}"
            )
    lines.extend(["", "--- PARAMETER STABILITY AMONG PRACTICAL CANDIDATES ---"])
    lines.extend(parameter_stability_summary(analyzed))
    lines.extend(["", "--- TINY SAMPLE TRAPS REJECTED ---"])
    if tiny.empty:
        lines.append("No tiny-sample traps found.")
    else:
        lines.append("These were rejected because validation_trades < 8, even if their win rate was high.")
        for _, row in tiny.iterrows():
            lines.append(
                f"{row['name']} | original score {row['score']:.2f} | "
                f"val trades {int(row['validation_trades'])} | win {row['validation_win_rate']:.1f}%"
            )
    return "\n".join(lines)


def candidate_config_from_row(row: pd.Series, base_config: dict[str, Any]) -> dict[str, Any]:
    config = dict(base_config)
    config["watchlist"] = str(row["symbols"]).split()
    config["strategy"] = dict(base_config["strategy"])
    for column in PARAM_COLUMNS:
        if column == "cooldown_hours":
            continue
        value = row[column]
        config["strategy"][column] = int(value) if column == "min_confidence" else float(value)
    config["bot"] = dict(base_config["bot"])
    config["bot"]["cooldown_hours"] = int(row["cooldown_hours"])
    config["candidate_note"] = (
        "Review-only optimization candidate. Do not treat as final until rerun with run_backtest."
    )
    return config


def write_candidate_config(analyzed: pd.DataFrame, output_path: str | Path, base_config: dict[str, Any]) -> bool:
    top = top_practical_candidates(analyzed, limit=1)
    if top.empty:
        return False
    config = candidate_config_from_row(top.iloc[0], base_config)
    Path(output_path).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return True


def main() -> int:
    args = parse_args()
    analyzed = analyze_results(load_results(args.inputs))
    analyzed.to_csv(args.output, index=False)
    report = generate_report(analyzed, args.inputs)
    Path(args.report).write_text(report + "\n", encoding="utf-8")
    wrote_candidate = write_candidate_config(analyzed, args.candidate_config, load_config())

    print(report)
    print("")
    print(f"Analysis CSV written to {args.output}")
    print(f"Report written to {args.report}")
    if wrote_candidate:
        print(f"Review-only candidate config written to {args.candidate_config}")
    else:
        print("No candidate config written because no Practical Review candidate was found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
