"""Telegram queue consumer and command poller."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import socket
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import connection as urllib3_connection
from urllib3.util.retry import Retry

from bot.access_control import (
    add_admin,
    bootstrap_owner,
    create_access_code,
    active_recipient_chat_ids,
    is_active_user,
    is_admin,
    is_owner,
    kick_user,
    redeem_access_code,
    remove_admin,
    revoke_access_code,
)
from bot.formatter import format_signal, format_status, format_watchlist
from bot.scanner import current_biases
from bot.state_manager import StateManager

LOGGER = logging.getLogger(__name__)
TOKEN_IN_URL_PATTERN = re.compile(r"/bot[^/\s]+")


class TelegramClient:
    def __init__(self, token: str, chat_id: str, poll_timeout_seconds: int = 0, force_ipv4: bool = True):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.offset = 0
        self.poll_timeout_seconds = poll_timeout_seconds
        if force_ipv4:
            urllib3_connection.allowed_gai_family = lambda: socket.AF_INET
        self.session = requests.Session()
        retry = Retry(
            total=3,
            connect=3,
            read=2,
            status=2,
            backoff_factor=0.7,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset(["GET", "POST"]),
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def send_message(self, text: str, chat_id: str | int | None = None) -> None:
        response = self.session.post(
            f"{self.base_url}/sendMessage",
            json={"chat_id": str(chat_id or self.chat_id), "text": text},
            timeout=(10, 20),
        )
        response.raise_for_status()

    def get_updates(self) -> list[dict]:
        response = self.session.post(
            f"{self.base_url}/getUpdates",
            json={"timeout": self.poll_timeout_seconds, "offset": self.offset, "allowed_updates": ["message"]},
            timeout=(8, 15),
        )
        response.raise_for_status()
        payload = response.json()
        updates = payload.get("result", [])
        if updates:
            self.offset = max(update["update_id"] for update in updates) + 1
        return updates


def sanitize_telegram_error(exc: Exception) -> str:
    """Remove bot tokens from request exception messages before logging."""
    return TOKEN_IN_URL_PATTERN.sub("/bot<redacted>", str(exc))


async def consume_signals(signal_queue: asyncio.Queue, client: TelegramClient, state_manager: StateManager):
    while True:
        signal = await signal_queue.get()
        if state_manager.is_on_cooldown(signal.symbol, signal.direction):
            continue
        message = format_signal(signal)
        recipients = active_recipient_chat_ids(state_manager)
        sent_count = 0
        for chat_id in recipients:
            try:
                client.send_message(message, chat_id=chat_id)
                sent_count += 1
            except requests.RequestException as exc:
                LOGGER.warning("Telegram signal send failed for chat %s: %s", chat_id, sanitize_telegram_error(exc))
        if not recipients or sent_count == 0:
            await signal_queue.put(signal)
            await asyncio.sleep(30)
            continue
        state_manager.state.last_signal = {"message": message, "timestamp": signal.timestamp.isoformat()}
        state_manager.state.signals_today += 1
        state_manager.set_cooldown(signal.symbol, signal.direction)


def _chat_and_user(message: dict) -> tuple[str, str, str | None]:
    chat_id = str(message.get("chat", {}).get("id", ""))
    user = message.get("from", {})
    user_id = str(user.get("id") or chat_id)
    username = user.get("username")
    return chat_id, user_id, username


def _actor_context(state_manager: StateManager, message: dict) -> tuple[str, str, str | None, str, bool, bool, bool]:
    chat_id, user_id, username = _chat_and_user(message)
    chat_type = message.get("chat", {}).get("type", "private")
    private_chat_id = chat_id if chat_type == "private" else None
    actor_ids = [user_id]
    if private_chat_id and private_chat_id != user_id:
        actor_ids.append(private_chat_id)

    actor_is_active = any(is_active_user(state_manager, actor_id) for actor_id in actor_ids)
    actor_is_admin = any(is_admin(state_manager, actor_id) for actor_id in actor_ids)
    actor_is_owner = any(is_owner(state_manager, actor_id) for actor_id in actor_ids)

    actor_id = user_id
    for candidate_id in actor_ids:
        if is_owner(state_manager, candidate_id) or is_admin(state_manager, candidate_id) or is_active_user(
            state_manager, candidate_id
        ):
            actor_id = candidate_id
            break
    return chat_id, user_id, username, actor_id, actor_is_active, actor_is_admin, actor_is_owner


def _code_list_text(state_manager: StateManager) -> str:
    if not state_manager.state.access_codes:
        return "No access codes created yet."
    lines = ["Access codes:"]
    now = time.time()
    for code, entry in sorted(state_manager.state.access_codes.items()):
        if entry.get("revoked"):
            status = "revoked"
        elif entry.get("used_by"):
            status = f"used by {entry['used_by']}"
        elif float(entry.get("expires_at", 0)) < now:
            status = "expired"
        else:
            status = "active"
        lines.append(f"{code}: {status}")
    return "\n".join(lines)


def _users_text(state_manager: StateManager) -> str:
    if not state_manager.state.users:
        return "No users yet."
    lines = ["Users:"]
    for user_id, user in sorted(state_manager.state.users.items()):
        status = "active" if user.get("active") else "inactive"
        role = user.get("role", "user")
        username = f" @{user['username']}" if user.get("username") else ""
        lines.append(f"{user_id}: {status} {role}{username}")
    return "\n".join(lines)


def _admin_list_text(state_manager: StateManager) -> str:
    active = [
        f"{admin_id}: {admin.get('role', 'admin')}"
        for admin_id, admin in sorted(state_manager.state.admins.items())
        if admin.get("active")
    ]
    return "Admins:\n" + "\n".join(active) if active else "No active admins."


def _menu_text(actor_is_admin: bool = False, actor_is_owner: bool = False) -> str:
    lines = [
        "Bot menu",
        "",
        "Access",
        "/start CODE - unlock access with a one-time code",
        "/menu - show this command menu",
        "/help - show this command menu",
        "/id - show your Telegram user/chat ID",
        "",
        "User commands",
        "/status - show bot status",
        "/watchlist - show watched symbols and current bias",
        "/lastsignal - show the latest delivered signal",
    ]
    if actor_is_admin:
        lines.extend(
            [
                "",
                "Admin commands",
                "/code_create - create a 24-hour one-time access code",
                "/code_list - list access codes and usage status",
                "/code_revoke CODE - revoke an unused access code",
                "/users - list active and inactive users",
                "/kick USER_ID - deactivate a normal user",
                "/admin_list - list active admins",
            ]
        )
    if actor_is_owner:
        lines.extend(
            [
                "",
                "Owner-only commands",
                "/admin_add USER_ID - promote an active user to admin",
                "/admin_remove USER_ID - remove admin access",
            ]
        )
    return "\n".join(lines)


def handle_command(client: TelegramClient, state_manager: StateManager, config: dict, message: dict) -> None:
    text = message.get("text", "").strip()
    if not text:
        return
    chat_id, user_id, username, actor_id, actor_is_active, actor_is_admin, actor_is_owner = _actor_context(
        state_manager, message
    )
    parts = text.split()
    command = parts[0].split("@", 1)[0].lower()

    if command in {"/menu", "/help"}:
        if not (actor_is_active or actor_is_admin):
            client.send_message(_menu_text(), chat_id=chat_id)
            return
        client.send_message(_menu_text(actor_is_admin, actor_is_owner), chat_id=chat_id)
        return

    if command == "/id":
        role = "owner" if actor_is_owner else "admin" if actor_is_admin else "user" if actor_is_active else "unauthorized"
        client.send_message(f"Your user ID: {user_id}\nThis chat ID: {chat_id}\nRole: {role}", chat_id=chat_id)
        return

    if command == "/start":
        if actor_is_admin:
            role = "owner" if actor_is_owner else "admin"
            client.send_message(
                f"You are already authorized as {role}. Use /menu to see available commands.", chat_id=chat_id
            )
            return
        if actor_is_active:
            client.send_message("Access already active. You will receive signals.", chat_id=chat_id)
            return
        if len(parts) < 2:
            client.send_message("Private bot. Send /start CODE to request access.", chat_id=chat_id)
            return
        ok, response = redeem_access_code(state_manager, parts[1].strip().upper(), user_id, chat_id, username)
        client.send_message(response, chat_id=chat_id)
        return

    if not (actor_is_active or actor_is_admin):
        client.send_message("Access denied. Ask an admin for a one-time access code.", chat_id=chat_id)
        return

    if command == "/status":
        client.send_message(format_status(state_manager.state, config), chat_id=chat_id)
    elif command == "/watchlist":
        client.send_message(format_watchlist(current_biases(config)), chat_id=chat_id)
    elif command == "/lastsignal":
        last_signal = state_manager.state.last_signal
        client.send_message(last_signal["message"] if last_signal else "No signals yet.", chat_id=chat_id)
    elif command == "/code_create":
        if not actor_is_admin:
            client.send_message("Admin only.", chat_id=chat_id)
            return
        code = create_access_code(state_manager, actor_id)
        client.send_message(f"Access code created: {code}\nExpires in 24 hours. One-time use.", chat_id=chat_id)
    elif command == "/code_list":
        if not actor_is_admin:
            client.send_message("Admin only.", chat_id=chat_id)
            return
        client.send_message(_code_list_text(state_manager), chat_id=chat_id)
    elif command == "/code_revoke":
        if not actor_is_admin:
            client.send_message("Admin only.", chat_id=chat_id)
            return
        if len(parts) < 2:
            client.send_message("Usage: /code_revoke CODE", chat_id=chat_id)
            return
        revoked = revoke_access_code(state_manager, parts[1].strip().upper(), actor_id)
        client.send_message("Code revoked." if revoked else "Code not found.", chat_id=chat_id)
    elif command == "/users":
        if not actor_is_admin:
            client.send_message("Admin only.", chat_id=chat_id)
            return
        client.send_message(_users_text(state_manager), chat_id=chat_id)
    elif command == "/kick":
        if not actor_is_admin:
            client.send_message("Admin only.", chat_id=chat_id)
            return
        if len(parts) < 2:
            client.send_message("Usage: /kick USER_ID", chat_id=chat_id)
            return
        kicked = kick_user(state_manager, parts[1], actor_id)
        client.send_message("User kicked." if kicked else "User not found or cannot be kicked.", chat_id=chat_id)
    elif command == "/admin_list":
        if not actor_is_admin:
            client.send_message("Admin only.", chat_id=chat_id)
            return
        client.send_message(_admin_list_text(state_manager), chat_id=chat_id)
    elif command == "/admin_add":
        if not actor_is_owner:
            client.send_message("Owner only.", chat_id=chat_id)
            return
        if len(parts) < 2:
            client.send_message("Usage: /admin_add USER_ID", chat_id=chat_id)
            return
        added = add_admin(state_manager, parts[1], actor_id)
        client.send_message("Admin added." if added else "User must be active before becoming admin.", chat_id=chat_id)
    elif command == "/admin_remove":
        if not actor_is_owner:
            client.send_message("Owner only.", chat_id=chat_id)
            return
        if len(parts) < 2:
            client.send_message("Usage: /admin_remove USER_ID", chat_id=chat_id)
            return
        removed = remove_admin(state_manager, parts[1], actor_id)
        client.send_message("Admin removed." if removed else "Admin not found or cannot be removed.", chat_id=chat_id)
    else:
        client.send_message("Unknown command.", chat_id=chat_id)


async def poll_commands(client: TelegramClient, state_manager: StateManager, config: dict):
    failure_count = 0
    while True:
        try:
            updates = client.get_updates()
            if updates:
                LOGGER.info("Telegram polling received %s update(s)", len(updates))
            for update in updates:
                message = update.get("message", {})
                handle_command(client, state_manager, config, message)
            failure_count = 0
        except requests.RequestException as exc:
            failure_count += 1
            backoff_seconds = min(30, 5 * failure_count)
            LOGGER.warning(
                "Telegram polling failed; retrying in %s seconds: %s",
                backoff_seconds,
                sanitize_telegram_error(exc),
            )
            await asyncio.sleep(backoff_seconds)
            continue
        await asyncio.sleep(2)


async def run(signal_queue: asyncio.Queue, config: dict, state_manager: StateManager):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise RuntimeError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set in environment or .env")
    bootstrap_owner(state_manager, chat_id)
    client = TelegramClient(token, chat_id)
    tasks = [consume_signals(signal_queue, client, state_manager)]
    if config.get("bot", {}).get("enable_telegram_commands", True):
        tasks.append(poll_commands(client, state_manager, config))
        LOGGER.info("Telegram command polling enabled")
    else:
        LOGGER.info("Telegram command polling disabled; signal delivery remains enabled")
    await asyncio.gather(*tasks)
