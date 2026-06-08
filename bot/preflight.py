"""Bounded live-bot preflight checks."""

from __future__ import annotations

import argparse
import asyncio
import os

from backtest.data_fetcher import check_data_source, set_data_source
from backtest.strategy import load_config
from bot.main import load_environment
from bot.scanner import scan_once
from bot.telegram_client import TelegramClient


def _require_env() -> tuple[str, str]:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise RuntimeError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set in environment or .env")
    return token, chat_id


async def run_preflight(send_test_message: bool = False, scan: bool = True) -> int:
    """Run finite checks for env, Bybit access, optional scan, and optional Telegram send."""
    load_environment()
    config = load_config()
    token, chat_id = _require_env()

    bybit_base_url = config["backtest"].get("bybit_api_base_url")
    set_data_source(bybit_base_url, config["backtest"].get("cache_dir"))

    print("Preflight")
    print("  env: OK")
    print(f"  bybit endpoint: {bybit_base_url}")
    if not check_data_source():
        print("  bybit: FAILED")
        return 1
    print("  bybit: OK")

    emitted = 0
    if scan:
        queue: asyncio.Queue = asyncio.Queue()
        emitted = await scan_once(queue, config)
        print(f"  scan_once: OK ({emitted} latest-candle signals)")

    if send_test_message:
        client = TelegramClient(token, chat_id)
        client.send_message(
            "Futures Signal Bot preflight OK\n"
            f"Bybit: OK\n"
            f"Latest-candle signals found: {emitted}"
        )
        print("  telegram send: OK")

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run finite live-bot preflight checks.")
    parser.add_argument(
        "--send-test-message",
        action="store_true",
        help="Send a Telegram confirmation message after env, Bybit, and scan checks pass.",
    )
    parser.add_argument(
        "--no-scan",
        action="store_true",
        help="Skip the one-shot market scan and only verify env plus Bybit access.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return asyncio.run(run_preflight(send_test_message=args.send_test_message, scan=not args.no_scan))


if __name__ == "__main__":
    raise SystemExit(main())
