import pytest

from bot.preflight import _require_env, run_preflight


def test_require_env_rejects_missing_secrets(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)

    with pytest.raises(RuntimeError, match="TELEGRAM_BOT_TOKEN"):
        _require_env()


@pytest.mark.asyncio
async def test_run_preflight_no_scan(monkeypatch, capsys):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "secret-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123")
    monkeypatch.setattr("bot.preflight.load_environment", lambda: None)
    monkeypatch.setattr(
        "bot.preflight.load_config",
        lambda: {"backtest": {"bybit_api_base_url": "https://example.test", "cache_dir": "cache"}},
    )
    monkeypatch.setattr("bot.preflight.set_data_source", lambda *args: None)
    monkeypatch.setattr("bot.preflight.check_data_source", lambda: True)

    exit_code = await run_preflight(scan=False)

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "env: OK" in output
    assert "bybit: OK" in output
    assert "scan_once" not in output
    assert "secret-token" not in output


@pytest.mark.asyncio
async def test_run_preflight_scan_and_send(monkeypatch, capsys):
    sent_messages = []

    class FakeTelegramClient:
        def __init__(self, token, chat_id):
            self.token = token
            self.chat_id = chat_id

        def send_message(self, text):
            sent_messages.append(text)

    async def fake_scan_once(queue, config):
        return 2

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "secret-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123")
    monkeypatch.setattr("bot.preflight.load_environment", lambda: None)
    monkeypatch.setattr(
        "bot.preflight.load_config",
        lambda: {
            "watchlist": ["BTCUSDT"],
            "strategy": {"entry_timeframe": "15", "trend_timeframe": "60"},
            "bot": {"cooldown_hours": 4},
            "backtest": {"bybit_api_base_url": "https://example.test", "cache_dir": "cache"},
        },
    )
    monkeypatch.setattr("bot.preflight.set_data_source", lambda *args: None)
    monkeypatch.setattr("bot.preflight.check_data_source", lambda: True)
    monkeypatch.setattr("bot.preflight.scan_once", fake_scan_once)
    monkeypatch.setattr("bot.preflight.TelegramClient", FakeTelegramClient)

    exit_code = await run_preflight(send_test_message=True)

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "scan_once: OK (2 latest-candle signals)" in output
    assert "telegram send: OK" in output
    assert sent_messages == [
        "Futures Signal Bot preflight OK\nBybit: OK\nLatest-candle signals found: 2"
    ]


@pytest.mark.asyncio
async def test_run_preflight_fails_when_bybit_check_fails(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "secret-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123")
    monkeypatch.setattr("bot.preflight.load_environment", lambda: None)
    monkeypatch.setattr(
        "bot.preflight.load_config",
        lambda: {"backtest": {"bybit_api_base_url": "https://example.test", "cache_dir": "cache"}},
    )
    monkeypatch.setattr("bot.preflight.set_data_source", lambda *args: None)
    monkeypatch.setattr("bot.preflight.check_data_source", lambda: False)

    assert await run_preflight(scan=False) == 1
