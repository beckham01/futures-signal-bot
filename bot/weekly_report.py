"""CLI wrapper: run the weekly telemetry report and alert admins on Telegram
if any strategy's rolling win rate has dropped too far below its backtested
target. Core aggregation/alerting logic lives in backtest/weekly_report.py;
this module only adds the live Telegram delivery.

Usage:
    python -m bot.weekly_report
    python -m bot.weekly_report --telemetry-path logs/paper_trade_telemetry.csv
"""

from __future__ import annotations

import argparse
import logging
import os

from backtest.strategy import load_config
from backtest.weekly_report import build_weekly_report
from bot.access_control import active_admin_chat_ids
from bot.main import load_environment
from bot.state_manager import StateManager
from bot.telegram_client import TelegramClient

LOGGER = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Weekly telemetry report with Telegram alerting.")
    parser.add_argument("--telemetry-path", default="logs/live_trade_telemetry.csv")
    parser.add_argument("--config", default="config.yaml")
    return parser.parse_args()


def send_admin_alerts(warnings: list[str], config: dict) -> None:
    if not warnings:
        return
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        LOGGER.warning("Cannot send Telegram alerts: TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID not set")
        return
    state_manager = StateManager(config["bot"]["state_file"], config["bot"]["cooldown_hours"])
    client = TelegramClient(token, chat_id)
    message = "Weekly report alert:\n" + "\n".join(warnings)
    for admin_chat_id in active_admin_chat_ids(state_manager):
        try:
            client.send_message(message, chat_id=admin_chat_id)
        except Exception as exc:  # noqa: BLE001 - best-effort alert delivery
            LOGGER.warning("Failed to send weekly report alert to %s: %s", admin_chat_id, exc)


def main() -> int:
    load_environment()
    args = _parse_args()
    config = load_config(args.config)
    summary, warnings = build_weekly_report(args.telemetry_path)
    print("=== WEEKLY TELEMETRY REPORT ===")
    if summary.empty:
        print("No resolved trades yet.")
    else:
        print(summary.to_string(index=False))
    if warnings:
        print("\n--- ALERTS ---")
        for warning in warnings:
            print(f"WARNING: {warning}")
    send_admin_alerts(warnings, config)
    return 1 if warnings else 0


if __name__ == "__main__":
    raise SystemExit(main())
