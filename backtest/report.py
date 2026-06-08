"""Human-readable backtest report generation."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from backtest.simulator import TradeResult


def _pct(value: int, total: int) -> float:
    return (value / total * 100) if total else 0.0


def _max_consecutive_losses(results: list[TradeResult]) -> int:
    longest = 0
    current = 0
    for result in sorted(results, key=lambda item: item.signal.timestamp):
        if result.outcome == "STOP_HIT":
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _group_metrics(results: list[TradeResult]) -> tuple[int, float, float]:
    count = len(results)
    wins = sum(1 for result in results if result.outcome in {"TP1_ONLY", "TP2_HIT"})
    avg_r = sum(result.pnl_r for result in results) / count if count else 0.0
    return count, _pct(wins, count), avg_r


def _total_r(results: list[TradeResult]) -> float:
    return sum(result.pnl_r for result in results)


def _trades_per_week(results: list[TradeResult], days: int) -> float:
    return len(results) / (days / 7) if days else 0.0


def _top_symbol_share(results: list[TradeResult]) -> float:
    if not results:
        return 0.0
    counts = Counter(result.signal.symbol for result in results)
    return max(counts.values()) / len(results)


def strategy_b_acceptance(results: list[TradeResult], days: int) -> tuple[bool, list[str]]:
    count, win_rate, avg_r = _group_metrics(results)
    trades_week = _trades_per_week(results, days)
    total_r = _total_r(results)
    max_losses = _max_consecutive_losses(results)
    top_share = _top_symbol_share(results)
    failures = []
    if trades_week < 2.5 or trades_week > 9:
        failures.append(f"trades/week {trades_week:.2f} outside 2.5-9")
    if win_rate < 40:
        failures.append(f"win rate {win_rate:.1f}% < 40%")
    if total_r <= 0:
        failures.append(f"total R {total_r:.2f} <= 0")
    if avg_r <= 0:
        failures.append(f"avg R {avg_r:.2f} <= 0")
    if max_losses > 6:
        failures.append(f"max consecutive losses {max_losses} > 6")
    if top_share > 0.60:
        failures.append(f"top symbol share {top_share:.2f} > 0.60")
    return not failures, failures


def combined_acceptance(results_a: list[TradeResult], combined: list[TradeResult], days: int) -> tuple[bool, list[str]]:
    _, win_rate, avg_r = _group_metrics(combined)
    trades_week_a = _trades_per_week(results_a, days)
    trades_week_combined = _trades_per_week(combined, days)
    total_r_a = _total_r(results_a)
    total_r_combined = _total_r(combined)
    failures = []
    if trades_week_combined < trades_week_a + 2:
        failures.append(f"combined trades/week {trades_week_combined:.2f} not at least A + 2")
    if total_r_combined < total_r_a:
        failures.append(f"combined total R {total_r_combined:.2f} < A total R {total_r_a:.2f}")
    if avg_r <= 0:
        failures.append(f"combined avg R {avg_r:.2f} <= 0")
    if win_rate < 38:
        failures.append(f"combined win rate {win_rate:.1f}% < 38%")
    return not failures, failures


def confidence_bucket(score: int) -> str:
    """Return a narrow score bucket for confidence diagnostics."""
    if score >= 95:
        return "95-100"
    if score >= 85:
        return "85-94"
    if score >= 75:
        return "75-84"
    if score >= 65:
        return "65-74"
    if score >= 55:
        return "55-64"
    return "<55"


def summarize_by_confidence_condition(results: list[TradeResult]) -> dict[str, tuple[int, float, float]]:
    """Return metrics for each true confidence condition."""
    grouped: dict[str, list[TradeResult]] = defaultdict(list)
    for result in results:
        for condition, enabled in result.signal.confidence_conditions.items():
            if enabled:
                grouped[condition].append(result)
    return {condition: _group_metrics(items) for condition, items in grouped.items()}


def generate_report(
    results: list[TradeResult],
    symbols: list[str],
    start_date: str,
    end_date: str,
    filtered_confidence: int = 0,
    filtered_rr: int = 0,
    output_path: str | Path = "backtest_report.txt",
) -> str:
    """Print and save a backtest report."""
    total = len(results)
    outcomes = Counter(result.outcome for result in results)
    longs = sum(1 for result in results if result.signal.direction == "LONG")
    shorts = total - longs
    wins = outcomes["TP1_ONLY"] + outcomes["TP2_HIT"]
    avg_r = sum(result.pnl_r for result in results) / total if total else 0.0
    total_r = sum(result.pnl_r for result in results)
    avg_bars = sum(result.bars_held for result in results) / total if total else 0.0

    by_symbol: dict[str, list[TradeResult]] = defaultdict(list)
    by_label: dict[str, list[TradeResult]] = defaultdict(list)
    by_direction: dict[str, list[TradeResult]] = defaultdict(list)
    by_month: dict[str, list[TradeResult]] = defaultdict(list)
    by_bucket: dict[str, list[TradeResult]] = defaultdict(list)
    for result in results:
        by_symbol[result.signal.symbol].append(result)
        by_label[result.signal.confidence_label].append(result)
        by_direction[result.signal.direction].append(result)
        by_month[result.signal.timestamp.strftime("%Y-%m")].append(result)
        by_bucket[confidence_bucket(result.signal.confidence)].append(result)

    lines = [
        "=== BACKTEST REPORT ===",
        f"Period: {start_date} to {end_date}",
        f"Symbols: {', '.join(symbols)}",
        "",
        "--- SIGNAL SUMMARY ---",
        f"Total signals generated: {total + filtered_confidence + filtered_rr}",
        f"  Long signals: {longs}",
        f"  Short signals: {shorts}",
        f"  Filtered by confidence (<55): {filtered_confidence}",
        f"  Filtered by R:R (<3): {filtered_rr}",
        f"  Final signals evaluated: {total}",
        "",
        "--- OUTCOME DISTRIBUTION ---",
        f"TP2 hit (full winner):     {outcomes['TP2_HIT']}  ({_pct(outcomes['TP2_HIT'], total):.0f}%)",
        f"TP1 only (breakeven+):     {outcomes['TP1_ONLY']}  ({_pct(outcomes['TP1_ONLY'], total):.0f}%)",
        f"Stop hit (full loss):      {outcomes['STOP_HIT']}  ({_pct(outcomes['STOP_HIT'], total):.0f}%)",
        f"Open / unresolved:         {outcomes['OPEN']}  ({_pct(outcomes['OPEN'], total):.0f}%)",
        "",
        "--- PERFORMANCE METRICS ---",
        f"Win rate (TP1 or better):  {_pct(wins, total):.0f}%",
        f"Full win rate (TP2 hit):   {_pct(outcomes['TP2_HIT'], total):.0f}%",
        f"Average R per trade:       {avg_r:.2f}",
        f"Total R if 1% risk/trade:  {total_r:.2f}%",
        f"Max consecutive losses:    {_max_consecutive_losses(results)}",
        f"Average bars held:         {avg_bars:.0f} ({avg_bars * 0.25:.1f}h)",
        "",
        "--- BY SYMBOL ---",
    ]
    for symbol in symbols:
        count, win_rate, symbol_avg_r = _group_metrics(by_symbol.get(symbol, []))
        lines.append(f"{symbol}: {count} trades | Win rate {win_rate:.0f}% | Avg R {symbol_avg_r:.2f}")

    lines.extend(["", "--- BY DIRECTION ---"])
    for direction in ["LONG", "SHORT"]:
        count, win_rate, direction_avg_r = _group_metrics(by_direction.get(direction, []))
        lines.append(f"{direction}: {count} trades | Win rate {win_rate:.0f}% | Avg R {direction_avg_r:.2f}")

    lines.extend(["", "--- BY MONTH ---"])
    for month in sorted(by_month):
        count, win_rate, month_avg_r = _group_metrics(by_month[month])
        lines.append(f"{month}: {count} trades | Win rate {win_rate:.0f}% | Avg R {month_avg_r:.2f}")

    lines.extend(["", "--- BY CONFIDENCE LABEL ---"])
    for label in ["STRONG", "HIGH", "MODERATE"]:
        count, win_rate, label_avg_r = _group_metrics(by_label.get(label, []))
        label_display = {"STRONG": "STRONG  (85+)", "HIGH": "HIGH   (70-84)", "MODERATE": "MODERATE(55-69)"}[label]
        lines.append(f"{label_display}: {count} trades | Win rate {win_rate:.0f}% | Avg R {label_avg_r:.2f}")

    lines.extend(["", "--- BY CONFIDENCE SCORE BUCKET ---"])
    for bucket in ["95-100", "85-94", "75-84", "65-74", "55-64", "<55"]:
        count, win_rate, bucket_avg_r = _group_metrics(by_bucket.get(bucket, []))
        lines.append(f"{bucket}: {count} trades | Win rate {win_rate:.0f}% | Avg R {bucket_avg_r:.2f}")

    lines.extend(["", "--- BY CONFIDENCE CONDITION TRUE ---"])
    condition_summary = summarize_by_confidence_condition(results)
    for condition in sorted(condition_summary):
        count, win_rate, condition_avg_r = condition_summary[condition]
        lines.append(f"{condition}: {count} trades | Win rate {win_rate:.0f}% | Avg R {condition_avg_r:.2f}")

    lines.extend(["", "--- SIGNAL LOG (last 20) ---"])
    for result in sorted(results, key=lambda item: item.signal.timestamp)[-20:]:
        lines.append(
            f"{result.signal.timestamp} | {result.signal.symbol} | {result.signal.direction} | "
            f"Confidence: {result.signal.confidence} | Outcome: {result.outcome} | R: {result.pnl_r:.2f}"
        )

    report = "\n".join(lines)
    Path(output_path).write_text(report + "\n", encoding="utf-8")
    print(report)
    return report


def generate_report_b(
    results: list[TradeResult],
    start_date: str,
    end_date: str,
    symbols: list[str],
    days: int,
    output_path: str | Path = "strategy_b_report.txt",
) -> str:
    """Print and save a Strategy B report with acceptance verdict."""
    report = generate_report(results, symbols, start_date, end_date, output_path=output_path)
    passed, failures = strategy_b_acceptance(results, days)
    verdict = [
        "",
        "--- STRATEGY B ACCEPTANCE ---",
        f"Trades/week: {_trades_per_week(results, days):.2f}",
        f"Top symbol share: {_top_symbol_share(results):.2f}",
        f"Verdict: {'PASS' if passed else 'FAIL'}",
        f"Reason: {'All Strategy B gates passed.' if passed else '; '.join(failures)}",
    ]
    full_report = report + "\n" + "\n".join(verdict)
    Path(output_path).write_text(full_report + "\n", encoding="utf-8")
    print("\n".join(verdict))
    return full_report


def _summary_line(label: str, results: list[TradeResult], days: int) -> str:
    count, win_rate, avg_r = _group_metrics(results)
    return (
        f"{label:<12} {count:>7} {win_rate:>8.1f}% {avg_r:>8.2f} "
        f"{_total_r(results):>8.2f} {_trades_per_week(results, days):>8.2f} "
        f"{_max_consecutive_losses(results):>8}"
    )


def _symbol_concentration(results: list[TradeResult]) -> str:
    if not results:
        return "none"
    counts = Counter(result.signal.symbol for result in results)
    total = len(results)
    return " | ".join(f"{symbol} {count / total * 100:.0f}%" for symbol, count in counts.most_common())


def generate_comparison_report(
    results_a: list[TradeResult],
    results_b: list[TradeResult],
    combined_results: list[TradeResult],
    conflicts: list[tuple],
    start_date: str,
    end_date: str,
    days: int,
    output_path: str | Path = "strategy_comparison_report.txt",
) -> str:
    """Print and save Strategy A/B comparison report."""
    passed_b, failures_b = strategy_b_acceptance(results_b, days)
    passed_combined, failures_combined = combined_acceptance(results_a, combined_results, days)
    delta_trades = _trades_per_week(combined_results, days) - _trades_per_week(results_a, days)
    delta_r = _total_r(combined_results) - _total_r(results_a)
    lines = [
        "=== STRATEGY COMPARISON REPORT ===",
        f"Period: {start_date} to {end_date}",
        "",
        f"{'Strategy':<12} {'Trades':>7} {'Win':>9} {'Avg R':>8} {'Total R':>8} {'Tr/Wk':>8} {'MaxLoss':>8}",
        "-" * 72,
        _summary_line("Strategy A", results_a, days),
        _summary_line("Strategy B", results_b, days),
        _summary_line("Combined", combined_results, days),
        "-" * 72,
        f"Conflicts detected: {len(conflicts)} (Strategy A took priority in all cases)",
        "",
        "--- SYMBOL CONCENTRATION ---",
        f"Strategy A: {_symbol_concentration(results_a)}",
        f"Strategy B: {_symbol_concentration(results_b)}",
        f"Combined: {_symbol_concentration(combined_results)}",
        "",
        "--- VERDICT ---",
        f"Strategy B acceptance: {'PASS' if passed_b else 'FAIL'}",
        f"Strategy B reason: {'All Strategy B gates passed.' if passed_b else '; '.join(failures_b)}",
        f"Combined acceptance: {'PASS' if passed_combined else 'FAIL'}",
        f"Combined reason: {'All combined gates passed.' if passed_combined else '; '.join(failures_combined)}",
        f"Combined improvement over A alone: {delta_trades:+.2f} trades/week | R delta: {delta_r:+.2f}",
    ]
    report = "\n".join(lines)
    Path(output_path).write_text(report + "\n", encoding="utf-8")
    print(report)
    return report
