import asyncio
import logging
import os

from bot.main import load_environment, main


def test_load_environment_reads_workspace_and_project_env(tmp_path, monkeypatch):
    workspace = tmp_path
    project = workspace / "futures-signal-bot"
    bot_dir = project / "bot"
    bot_dir.mkdir(parents=True)
    (workspace / ".env").write_text("TELEGRAM_BOT_TOKEN=workspace\n", encoding="utf-8")
    (project / ".env").write_text("TELEGRAM_BOT_TOKEN=project\nTELEGRAM_CHAT_ID=123\n", encoding="utf-8")

    fake_main = bot_dir / "main.py"
    monkeypatch.setattr("bot.main.__file__", str(fake_main))
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)

    load_environment()

    assert os.environ["TELEGRAM_BOT_TOKEN"] == "project"
    assert os.environ["TELEGRAM_CHAT_ID"] == "123"


def test_main_quiets_urllib3_connectionpool_logger(monkeypatch, tmp_path):
    async def fake_scan_loop(queue, config):
        await asyncio.Event().wait()

    async def fake_telegram_run(queue, config, state_manager):
        await asyncio.Event().wait()

    class Loop:
        def add_signal_handler(self, *args):
            raise NotImplementedError

    async def fake_wait(self):
        return None

    monkeypatch.setattr("bot.main.load_environment", lambda: None)
    monkeypatch.setattr(
        "bot.main.load_config",
        lambda: {
            "watchlist": ["BTCUSDT"],
            "env": {"LOG_LEVEL": "INFO"},
            "bot": {"state_file": str(tmp_path / "state.json"), "cooldown_hours": 4},
        },
    )
    monkeypatch.setattr("bot.main.scan_loop", fake_scan_loop)
    monkeypatch.setattr("bot.main.telegram_run", fake_telegram_run)
    monkeypatch.setattr("bot.main.asyncio.get_running_loop", lambda: Loop())
    monkeypatch.setattr("bot.main.asyncio.Event.wait", fake_wait)

    asyncio.run(main())

    assert logging.getLogger("urllib3.connectionpool").level == logging.ERROR
