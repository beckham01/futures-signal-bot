import os

from bot.main import load_environment


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
