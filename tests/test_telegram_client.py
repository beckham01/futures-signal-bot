import asyncio

import pytest
import requests

from bot.state_manager import StateManager
from bot.access_control import bootstrap_owner, create_access_code, redeem_access_code
from bot.telegram_client import TelegramClient, consume_signals, handle_command, poll_commands, run, sanitize_telegram_error


def test_telegram_run_requires_secrets(monkeypatch, tmp_path):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    manager = StateManager(str(tmp_path / "state.json"))
    with pytest.raises(RuntimeError, match="TELEGRAM_BOT_TOKEN"):
        asyncio.run(run(asyncio.Queue(), {"watchlist": []}, manager))


@pytest.mark.asyncio
async def test_run_can_disable_command_polling(monkeypatch, tmp_path):
    calls = []

    class FakeTelegramClient:
        def __init__(self, token, chat_id):
            self.token = token
            self.chat_id = chat_id

    async def fake_consume(queue, client, manager):
        calls.append("consume")

    async def fake_poll(client, manager, config):
        calls.append("poll")

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "chat")
    monkeypatch.setattr("bot.telegram_client.TelegramClient", FakeTelegramClient)
    monkeypatch.setattr("bot.telegram_client.consume_signals", fake_consume)
    monkeypatch.setattr("bot.telegram_client.poll_commands", fake_poll)
    manager = StateManager(str(tmp_path / "state.json"))

    await run(asyncio.Queue(), {"bot": {"enable_telegram_commands": False}}, manager)

    assert calls == ["consume"]


@pytest.mark.asyncio
async def test_run_enables_command_polling_by_default(monkeypatch, tmp_path):
    calls = []

    class FakeTelegramClient:
        def __init__(self, token, chat_id):
            self.token = token
            self.chat_id = chat_id

    async def fake_consume(queue, client, manager):
        calls.append("consume")

    async def fake_poll(client, manager, config):
        calls.append("poll")

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "chat")
    monkeypatch.setattr("bot.telegram_client.TelegramClient", FakeTelegramClient)
    monkeypatch.setattr("bot.telegram_client.consume_signals", fake_consume)
    monkeypatch.setattr("bot.telegram_client.poll_commands", fake_poll)
    manager = StateManager(str(tmp_path / "state.json"))

    await run(asyncio.Queue(), {"bot": {}}, manager)

    assert calls == ["consume", "poll"]


def test_sanitize_telegram_error_redacts_bot_token():
    message = (
        "HTTPSConnectionPool(host='api.telegram.org', port=443): "
        "Max retries exceeded with url: /bot123456:SECRET/getUpdates?timeout=3"
    )

    sanitized = sanitize_telegram_error(RuntimeError(message))

    assert "SECRET" not in sanitized
    assert "/bot<redacted>/getUpdates" in sanitized


def test_get_updates_uses_short_polling_post_and_advances_offset():
    calls = []

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"result": [{"update_id": 10, "message": {"text": "/menu"}}]}

    class Session:
        def post(self, url, json=None, timeout=None):
            calls.append((url, json, timeout))
            return Response()

    client = TelegramClient("token", "100")
    client.session = Session()

    updates = client.get_updates()

    assert updates[0]["update_id"] == 10
    assert client.offset == 11
    assert calls == [
        (
            "https://api.telegram.org/bottoken/getUpdates",
            {"timeout": 0, "offset": 0, "allowed_updates": ["message"]},
            (8, 15),
        )
    ]


def test_telegram_client_mounts_retry_adapters():
    client = TelegramClient("token", "100", force_ipv4=False)

    https_adapter = client.session.get_adapter("https://api.telegram.org")

    assert https_adapter.max_retries.total == 3
    assert https_adapter.max_retries.connect == 3


@pytest.mark.asyncio
async def test_consume_signals_requeues_on_send_failure(monkeypatch, tmp_path):
    class Signal:
        symbol = "BTCUSDT"
        direction = "LONG"
        timestamp = __import__("pandas").Timestamp("2026-01-01T00:00:00Z")

    class FailingClient:
        def send_message(self, text, chat_id=None):
            raise requests.ConnectTimeout("boom /botSECRET/sendMessage")

    queue = asyncio.Queue()
    await queue.put(Signal())
    manager = StateManager(str(tmp_path / "state.json"))
    bootstrap_owner(manager, "100")
    monkeypatch.setattr("bot.telegram_client.format_signal", lambda signal: "signal")

    task = asyncio.create_task(consume_signals(queue, FailingClient(), manager))
    await asyncio.sleep(0.05)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    assert queue.qsize() == 1
    assert manager.state.last_signal is None


@pytest.mark.asyncio
async def test_consume_signals_broadcasts_to_active_admin_and_user(monkeypatch, tmp_path):
    sent = []

    class Signal:
        symbol = "BTCUSDT"
        direction = "LONG"
        timestamp = __import__("pandas").Timestamp("2026-01-01T00:00:00Z")

    class Client:
        def send_message(self, text, chat_id=None):
            sent.append((chat_id, text))

    queue = asyncio.Queue()
    await queue.put(Signal())
    manager = StateManager(str(tmp_path / "state.json"))
    bootstrap_owner(manager, "100")
    code = create_access_code(manager, "100", code="USERCODE")
    redeem_access_code(manager, code, "200", "200")
    monkeypatch.setattr("bot.telegram_client.format_signal", lambda signal: "signal")

    task = asyncio.create_task(consume_signals(queue, Client(), manager))
    await asyncio.sleep(0.05)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    assert ("100", "signal") in sent
    assert ("200", "signal") in sent
    assert manager.state.signals_today == 1


@pytest.mark.asyncio
async def test_poll_commands_backs_off_on_failure(monkeypatch, tmp_path):
    sleeps = []

    class FailingClient:
        def get_updates(self):
            raise requests.ConnectTimeout("boom /botSECRET/getUpdates")

    async def fake_sleep(seconds):
        sleeps.append(seconds)
        raise asyncio.CancelledError

    monkeypatch.setattr("bot.telegram_client.asyncio.sleep", fake_sleep)
    manager = StateManager(str(tmp_path / "state.json"))

    with pytest.raises(asyncio.CancelledError):
        await poll_commands(FailingClient(), manager, {"watchlist": []})

    assert sleeps == [5]


def message(text, user_id="200", chat_id=None, username="alice", chat_type="private"):
    return {
        "text": text,
        "chat": {"id": chat_id or user_id, "type": chat_type},
        "from": {"id": user_id, "username": username},
    }


def test_handle_command_denies_unauthorized_status(tmp_path):
    sent = []

    class Client:
        def send_message(self, text, chat_id=None):
            sent.append((chat_id, text))

    manager = StateManager(str(tmp_path / "state.json"))
    bootstrap_owner(manager, "100")
    handle_command(Client(), manager, {"watchlist": []}, message("/status", "200"))
    assert sent == [("200", "Access denied. Ask an admin for a one-time access code.")]


def test_handle_command_start_activates_user(tmp_path):
    sent = []

    class Client:
        def send_message(self, text, chat_id=None):
            sent.append((chat_id, text))

    manager = StateManager(str(tmp_path / "state.json"))
    bootstrap_owner(manager, "100")
    create_access_code(manager, "100", code="VALIDCODE")
    handle_command(Client(), manager, {"watchlist": []}, message("/start VALIDCODE", "200"))
    assert manager.state.users["200"]["active"] is True
    assert "Access granted" in sent[-1][1]


def test_handle_command_start_for_owner_does_not_request_code(tmp_path):
    sent = []

    class Client:
        def send_message(self, text, chat_id=None):
            sent.append((chat_id, text))

    manager = StateManager(str(tmp_path / "state.json"))
    bootstrap_owner(manager, "100")
    handle_command(Client(), manager, {"watchlist": []}, message("/start", "100"))
    assert sent == [("100", "You are already authorized as owner. Use /menu to see available commands.")]


def test_handle_command_start_for_active_user_does_not_request_code(tmp_path):
    sent = []

    class Client:
        def send_message(self, text, chat_id=None):
            sent.append((chat_id, text))

    manager = StateManager(str(tmp_path / "state.json"))
    bootstrap_owner(manager, "100")
    code = create_access_code(manager, "100", code="USERCODE")
    redeem_access_code(manager, code, "200", "200")
    handle_command(Client(), manager, {"watchlist": []}, message("/start", "200"))
    assert sent == [("200", "Access already active. You will receive signals.")]


def test_handle_command_private_chat_owner_id_can_differ_from_user_id(tmp_path):
    sent = []

    class Client:
        def send_message(self, text, chat_id=None):
            sent.append(text)

    manager = StateManager(str(tmp_path / "state.json"))
    bootstrap_owner(manager, "100")
    handle_command(Client(), manager, {"watchlist": []}, message("/code_create", user_id="999", chat_id="100"))
    assert sent[-1].startswith("Access code created:")


def test_handle_command_group_chat_owner_id_does_not_grant_user_admin(tmp_path):
    sent = []

    class Client:
        def send_message(self, text, chat_id=None):
            sent.append(text)

    manager = StateManager(str(tmp_path / "state.json"))
    bootstrap_owner(manager, "100")
    handle_command(
        Client(),
        manager,
        {"watchlist": []},
        message("/code_create", user_id="999", chat_id="100", chat_type="group"),
    )
    assert sent == ["Access denied. Ask an admin for a one-time access code."]


def test_handle_command_menu_for_unauthorized_user_shows_public_commands(tmp_path):
    sent = []

    class Client:
        def send_message(self, text, chat_id=None):
            sent.append(text)

    manager = StateManager(str(tmp_path / "state.json"))
    bootstrap_owner(manager, "100")
    handle_command(Client(), manager, {"watchlist": []}, message("/menu", "200"))
    assert "/start CODE" in sent[-1]
    assert "/status" in sent[-1]
    assert "/code_create" not in sent[-1]


def test_handle_command_menu_for_admin_shows_admin_commands(tmp_path):
    sent = []

    class Client:
        def send_message(self, text, chat_id=None):
            sent.append(text)

    manager = StateManager(str(tmp_path / "state.json"))
    bootstrap_owner(manager, "100")
    handle_command(Client(), manager, {"watchlist": []}, message("/menu", "100"))
    assert "/code_create" in sent[-1]
    assert "/admin_add" in sent[-1]


def test_handle_command_help_aliases_menu(tmp_path):
    sent = []

    class Client:
        def send_message(self, text, chat_id=None):
            sent.append(text)

    manager = StateManager(str(tmp_path / "state.json"))
    bootstrap_owner(manager, "100")
    handle_command(Client(), manager, {"watchlist": []}, message("/help", "100"))
    assert sent[-1].startswith("🤖 Bot menu")


def test_handle_command_id_shows_ids_and_role(tmp_path):
    sent = []

    class Client:
        def send_message(self, text, chat_id=None):
            sent.append(text)

    manager = StateManager(str(tmp_path / "state.json"))
    bootstrap_owner(manager, "100")
    handle_command(Client(), manager, {"watchlist": []}, message("/id", user_id="999", chat_id="100"))
    assert "Your user ID: 999" in sent[-1]
    assert "This chat ID: 100" in sent[-1]
    assert "Role: owner" in sent[-1]


def test_handle_command_authorized_status(tmp_path):
    sent = []

    class Client:
        def send_message(self, text, chat_id=None):
            sent.append(text)

    manager = StateManager(str(tmp_path / "state.json"))
    bootstrap_owner(manager, "100")
    handle_command(Client(), manager, {"watchlist": ["BTCUSDT"]}, message("/status", "100"))
    assert "Bot online" in sent[0]


def test_handle_admin_code_and_user_commands(tmp_path):
    sent = []

    class Client:
        def send_message(self, text, chat_id=None):
            sent.append(text)

    manager = StateManager(str(tmp_path / "state.json"))
    bootstrap_owner(manager, "100")
    handle_command(Client(), manager, {"watchlist": []}, message("/code_create", "100"))
    generated_code = sent[-1].splitlines()[0].split(": ", 1)[1]
    handle_command(Client(), manager, {"watchlist": []}, message("/code_list", "100"))
    assert generated_code in sent[-1]
    handle_command(Client(), manager, {"watchlist": []}, message(f"/code_revoke {generated_code}", "100"))
    assert sent[-1] == "Code revoked."
    user_code = create_access_code(manager, "100", code="USERCODE")
    redeem_access_code(manager, user_code, "200", "200")
    handle_command(Client(), manager, {"watchlist": []}, message("/users", "100"))
    assert "200" in sent[-1]
    handle_command(Client(), manager, {"watchlist": []}, message("/kick 200", "100"))
    assert sent[-1] == "User kicked."


def test_handle_owner_admin_commands_and_non_owner_denial(tmp_path):
    sent = []

    class Client:
        def send_message(self, text, chat_id=None):
            sent.append(text)

    manager = StateManager(str(tmp_path / "state.json"))
    bootstrap_owner(manager, "100")
    code = create_access_code(manager, "100", code="ADMINCOD")
    redeem_access_code(manager, code, "200", "200")
    handle_command(Client(), manager, {"watchlist": []}, message("/admin_add 200", "100"))
    assert sent[-1] == "Admin added."
    handle_command(Client(), manager, {"watchlist": []}, message("/admin_add 300", "200"))
    assert sent[-1] == "Owner only."
    handle_command(Client(), manager, {"watchlist": []}, message("/admin_list", "200"))
    assert "200" in sent[-1]
    handle_command(Client(), manager, {"watchlist": []}, message("/admin_remove 200", "100"))
    assert sent[-1] == "Admin removed."
