"""Access-code and role management for the private Telegram bot."""

from __future__ import annotations

import secrets
import string
import time

from bot.state_manager import StateManager

OWNER_ROLE = "owner"
ADMIN_ROLE = "admin"
USER_ROLE = "user"
DEFAULT_CODE_TTL_SECONDS = 24 * 3600


def _id(value: int | str) -> str:
    return str(value)


def now_ts() -> float:
    return time.time()


def bootstrap_owner(state_manager: StateManager, owner_chat_id: int | str) -> None:
    """Ensure the configured TELEGRAM_CHAT_ID is the permanent owner/admin."""
    owner_id = _id(owner_chat_id)
    timestamp = now_ts()
    user = state_manager.state.users.get(owner_id, {})
    user.update(
        {
            "user_id": owner_id,
            "chat_id": owner_id,
            "role": OWNER_ROLE,
            "active": True,
            "joined_at": user.get("joined_at", timestamp),
        }
    )
    user.pop("revoked_at", None)
    state_manager.state.users[owner_id] = user
    state_manager.state.admins[owner_id] = {
        "user_id": owner_id,
        "role": OWNER_ROLE,
        "active": True,
        "added_at": state_manager.state.admins.get(owner_id, {}).get("added_at", timestamp),
        "added_by": owner_id,
    }
    state_manager.save()


def is_active_user(state_manager: StateManager, user_id: int | str) -> bool:
    user = state_manager.state.users.get(_id(user_id))
    return bool(user and user.get("active"))


def is_admin(state_manager: StateManager, user_id: int | str) -> bool:
    admin = state_manager.state.admins.get(_id(user_id))
    return bool(admin and admin.get("active"))


def is_owner(state_manager: StateManager, user_id: int | str) -> bool:
    admin = state_manager.state.admins.get(_id(user_id))
    return bool(admin and admin.get("active") and admin.get("role") == OWNER_ROLE)


def active_recipient_chat_ids(state_manager: StateManager) -> list[str]:
    """Return all active user/admin chat IDs once each."""
    chat_ids = []
    seen = set()
    for user_id, user in state_manager.state.users.items():
        if not user.get("active"):
            continue
        chat_id = str(user.get("chat_id", user_id))
        if chat_id in seen:
            continue
        seen.add(chat_id)
        chat_ids.append(chat_id)
    return chat_ids


def active_admin_chat_ids(state_manager: StateManager) -> list[str]:
    """Return active admin/owner chat IDs (for alerting, e.g. win-rate warnings)."""
    chat_ids = []
    seen = set()
    for admin_id, admin in state_manager.state.admins.items():
        if not admin.get("active"):
            continue
        user = state_manager.state.users.get(admin_id, {})
        chat_id = str(user.get("chat_id", admin_id))
        if chat_id in seen:
            continue
        seen.add(chat_id)
        chat_ids.append(chat_id)
    return chat_ids


def generate_code(length: int = 8) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def create_access_code(
    state_manager: StateManager,
    created_by: int | str,
    ttl_seconds: int = DEFAULT_CODE_TTL_SECONDS,
    code: str | None = None,
) -> str:
    if not is_admin(state_manager, created_by):
        raise PermissionError("Only admins can create access codes")
    code = code or generate_code()
    timestamp = now_ts()
    state_manager.state.access_codes[code] = {
        "code": code,
        "created_by": _id(created_by),
        "created_at": timestamp,
        "expires_at": timestamp + ttl_seconds,
        "used_by": None,
        "used_at": None,
        "revoked": False,
        "revoked_at": None,
    }
    state_manager.save()
    return code


def revoke_access_code(state_manager: StateManager, code: str, revoked_by: int | str) -> bool:
    if not is_admin(state_manager, revoked_by):
        raise PermissionError("Only admins can revoke access codes")
    entry = state_manager.state.access_codes.get(code)
    if not entry:
        return False
    entry["revoked"] = True
    entry["revoked_by"] = _id(revoked_by)
    entry["revoked_at"] = now_ts()
    state_manager.save()
    return True


def redeem_access_code(
    state_manager: StateManager,
    code: str,
    user_id: int | str,
    chat_id: int | str,
    username: str | None = None,
) -> tuple[bool, str]:
    user_key = _id(user_id)
    existing_user = state_manager.state.users.get(user_key)
    if existing_user and existing_user.get("active"):
        return True, "Access already active."
    entry = state_manager.state.access_codes.get(code)
    if not entry:
        return False, "Invalid access code."
    if entry.get("revoked"):
        return False, "Access code has been revoked."
    if entry.get("used_by"):
        return False, "Access code has already been used."
    if float(entry.get("expires_at", 0)) < now_ts():
        return False, "Access code has expired."

    timestamp = now_ts()
    state_manager.state.users[user_key] = {
        "user_id": user_key,
        "chat_id": _id(chat_id),
        "username": username,
        "role": USER_ROLE,
        "active": True,
        "joined_at": timestamp,
        "code": code,
    }
    entry["used_by"] = user_key
    entry["used_at"] = timestamp
    state_manager.save()
    return True, "Access granted. You will now receive bot signals."


def kick_user(state_manager: StateManager, user_id: int | str, kicked_by: int | str) -> bool:
    if not is_admin(state_manager, kicked_by):
        raise PermissionError("Only admins can kick users")
    user_key = _id(user_id)
    if is_owner(state_manager, user_key):
        return False
    if is_admin(state_manager, user_key) and not is_owner(state_manager, kicked_by):
        return False
    user = state_manager.state.users.get(user_key)
    if not user:
        return False
    user["active"] = False
    user["revoked_at"] = now_ts()
    user["revoked_by"] = _id(kicked_by)
    if user_key in state_manager.state.admins:
        state_manager.state.admins[user_key]["active"] = False
    state_manager.save()
    return True


def add_admin(state_manager: StateManager, user_id: int | str, added_by: int | str) -> bool:
    if not is_owner(state_manager, added_by):
        raise PermissionError("Only owner can add admins")
    user_key = _id(user_id)
    user = state_manager.state.users.get(user_key)
    if not user or not user.get("active"):
        return False
    user["role"] = ADMIN_ROLE
    state_manager.state.admins[user_key] = {
        "user_id": user_key,
        "role": ADMIN_ROLE,
        "active": True,
        "added_at": now_ts(),
        "added_by": _id(added_by),
    }
    state_manager.save()
    return True


def remove_admin(state_manager: StateManager, user_id: int | str, removed_by: int | str) -> bool:
    if not is_owner(state_manager, removed_by):
        raise PermissionError("Only owner can remove admins")
    user_key = _id(user_id)
    if is_owner(state_manager, user_key):
        return False
    admin = state_manager.state.admins.get(user_key)
    if not admin:
        return False
    admin["active"] = False
    user = state_manager.state.users.get(user_key)
    if user and user.get("active"):
        user["role"] = USER_ROLE
    state_manager.save()
    return True
