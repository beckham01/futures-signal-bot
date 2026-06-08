import time

import pytest

from bot.access_control import (
    add_admin,
    active_recipient_chat_ids,
    bootstrap_owner,
    create_access_code,
    is_admin,
    is_owner,
    kick_user,
    redeem_access_code,
    remove_admin,
    revoke_access_code,
)
from bot.state_manager import StateManager


def test_bootstrap_owner_from_env_chat_id(tmp_path):
    manager = StateManager(str(tmp_path / "state.json"))
    bootstrap_owner(manager, "100")
    assert is_owner(manager, "100")
    assert is_admin(manager, "100")
    assert active_recipient_chat_ids(manager) == ["100"]


def test_one_time_code_activates_exactly_one_user(tmp_path):
    manager = StateManager(str(tmp_path / "state.json"))
    bootstrap_owner(manager, "100")
    code = create_access_code(manager, "100", code="ABCD1234")
    ok, _ = redeem_access_code(manager, code, "200", "200", "alice")
    assert ok is True
    ok, message = redeem_access_code(manager, code, "201", "201", "bob")
    assert ok is False
    assert "already been used" in message


def test_used_expired_and_revoked_codes_are_rejected(tmp_path):
    manager = StateManager(str(tmp_path / "state.json"))
    bootstrap_owner(manager, "100")
    expired = create_access_code(manager, "100", ttl_seconds=-1, code="EXPIRED1")
    ok, message = redeem_access_code(manager, expired, "200", "200")
    assert ok is False
    assert "expired" in message

    revoked = create_access_code(manager, "100", code="REVOKED1")
    assert revoke_access_code(manager, revoked, "100") is True
    ok, message = redeem_access_code(manager, revoked, "201", "201")
    assert ok is False
    assert "revoked" in message


def test_kicked_users_stop_being_active(tmp_path):
    manager = StateManager(str(tmp_path / "state.json"))
    bootstrap_owner(manager, "100")
    code = create_access_code(manager, "100", code="KICKME1")
    redeem_access_code(manager, code, "200", "200")
    assert "200" in active_recipient_chat_ids(manager)
    assert kick_user(manager, "200", "100") is True
    assert "200" not in active_recipient_chat_ids(manager)


def test_owner_only_admin_management(tmp_path):
    manager = StateManager(str(tmp_path / "state.json"))
    bootstrap_owner(manager, "100")
    code = create_access_code(manager, "100", code="ADMIN001")
    redeem_access_code(manager, code, "200", "200")
    assert add_admin(manager, "200", "100") is True
    assert is_admin(manager, "200")

    with pytest.raises(PermissionError):
        add_admin(manager, "300", "200")
    assert remove_admin(manager, "200", "100") is True
    assert is_admin(manager, "200") is False
    assert remove_admin(manager, "100", "100") is False


def test_admin_can_create_revoke_codes_and_kick_users(tmp_path):
    manager = StateManager(str(tmp_path / "state.json"))
    bootstrap_owner(manager, "100")
    user_code = create_access_code(manager, "100", code="USER0001")
    redeem_access_code(manager, user_code, "200", "200")
    add_admin(manager, "200", "100")
    code = create_access_code(manager, "200", code="BYADMIN1")
    assert revoke_access_code(manager, code, "200") is True
    victim_code = create_access_code(manager, "200", code="VICTIM01")
    redeem_access_code(manager, victim_code, "300", "300")
    assert kick_user(manager, "300", "200") is True


def test_admin_cannot_kick_another_admin(tmp_path):
    manager = StateManager(str(tmp_path / "state.json"))
    bootstrap_owner(manager, "100")
    code_one = create_access_code(manager, "100", code="ADMIN001")
    code_two = create_access_code(manager, "100", code="ADMIN002")
    redeem_access_code(manager, code_one, "200", "200")
    redeem_access_code(manager, code_two, "300", "300")
    add_admin(manager, "200", "100")
    add_admin(manager, "300", "100")
    assert kick_user(manager, "300", "200") is False
    assert is_admin(manager, "300") is True
