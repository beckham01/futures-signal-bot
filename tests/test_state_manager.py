import logging

from bot.state_manager import BotState, StateManager


def test_state_manager_saves_and_loads(tmp_path):
    path = tmp_path / "state.json"
    manager = StateManager(str(path), cooldown_hours=4)
    manager.state.signals_today = 3
    manager.state.last_signal = {"message": "hello"}
    manager.save()

    reloaded = StateManager(str(path), cooldown_hours=4)
    assert reloaded.state.signals_today == 3
    assert reloaded.state.last_signal == {"message": "hello"}


def test_state_manager_cooldown(tmp_path):
    manager = StateManager(str(tmp_path / "state.json"), cooldown_hours=4)
    assert manager.is_on_cooldown("BTCUSDT", "LONG") is False
    manager.set_cooldown("BTCUSDT", "LONG")
    assert manager.is_on_cooldown("BTCUSDT", "LONG") is True


def test_state_manager_save_warning(tmp_path, caplog):
    manager = StateManager(str(tmp_path), cooldown_hours=4)
    with caplog.at_level(logging.WARNING):
        manager.save(BotState(cooldowns={}, last_signal=None, signals_today=0, last_scan_time=0.0))
    assert "Could not save bot state" in caplog.text
