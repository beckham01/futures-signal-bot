"""Live bot entry point."""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from pathlib import Path

from dotenv import load_dotenv

from backtest.strategy import load_config
from bot.scanner import scan_loop
from bot.state_manager import StateManager
from bot.telegram_client import run as telegram_run
from bot.weekly_report import scheduled_report_loop


def load_environment() -> None:
    """Load .env from the project directory and its parent workspace if present."""
    project_root = Path(__file__).resolve().parents[1]
    workspace_root = project_root.parent
    load_dotenv(workspace_root / ".env")
    load_dotenv(project_root / ".env", override=True)


async def main():
    load_environment()
    config = load_config()
    logging.basicConfig(
        level=config.get("env", {}).get("LOG_LEVEL", "INFO"),
        stream=sys.stdout,
        format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
    )
    logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)
    logging.getLogger(__name__).info("Live bot starting with %s symbols", len(config["watchlist"]))
    queue: asyncio.Queue = asyncio.Queue()
    state_manager = StateManager(config["bot"]["state_file"], config["bot"]["cooldown_hours"])

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            pass

    tasks = [
        asyncio.create_task(scan_loop(queue, config)),
        asyncio.create_task(telegram_run(queue, config, state_manager)),
        asyncio.create_task(scheduled_report_loop(config, state_manager)),
    ]
    await stop_event.wait()
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    state_manager.save()


if __name__ == "__main__":
    asyncio.run(main())
