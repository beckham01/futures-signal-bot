import asyncio

import pandas as pd
import pytest

from bot.access_control import bootstrap_owner
from bot.state_manager import StateManager
from bot.weekly_report import scheduled_report_loop, send_admin_alerts


def _state_manager(tmp_path) -> StateManager:
    state_manager = StateManager(str(tmp_path / "state.json"), cooldown_hours=4)
    bootstrap_owner(state_manager, "111")
    return state_manager


def _config(tmp_path) -> dict:
    return {"bot": {"state_file": str(tmp_path / "state.json"), "cooldown_hours": 4}}


def test_send_admin_alerts_noop_without_warnings(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr("bot.weekly_report.TelegramClient.send_message", lambda self, *a, **k: calls.append(a))
    send_admin_alerts([], _config(tmp_path))
    assert calls == []


def test_send_admin_alerts_noop_without_secrets(monkeypatch, tmp_path, caplog):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    caplog.set_level("WARNING")
    send_admin_alerts(["something bad"], _config(tmp_path))
    assert any("TELEGRAM_BOT_TOKEN" in record.message for record in caplog.records)


def test_send_admin_alerts_sends_to_active_admins(monkeypatch, tmp_path):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "111")
    state_manager = _state_manager(tmp_path)
    sent = []
    monkeypatch.setattr(
        "bot.weekly_report.TelegramClient.send_message",
        lambda self, text, chat_id=None: sent.append((chat_id, text)),
    )

    send_admin_alerts(["strategy_b win rate dropped"], _config(tmp_path), state_manager=state_manager)

    assert len(sent) == 1
    assert sent[0][0] == "111"
    assert "strategy_b win rate dropped" in sent[0][1]


@pytest.mark.asyncio
async def test_scheduled_report_loop_runs_periodically_and_never_crashes(monkeypatch, tmp_path):
    sleeps = []

    class StopLoop(Exception):
        pass

    async def fake_sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) >= 3:
            raise StopLoop

    build_calls = []

    def fake_build(path):
        build_calls.append(path)
        return pd.DataFrame(), []

    alert_calls = []

    monkeypatch.setattr("bot.weekly_report.asyncio.sleep", fake_sleep)
    monkeypatch.setattr("bot.weekly_report.build_weekly_report", fake_build)
    monkeypatch.setattr("bot.weekly_report.telemetry_path", lambda: "logs/live_trade_telemetry.csv")
    monkeypatch.setattr("bot.weekly_report.send_admin_alerts", lambda *args, **kwargs: alert_calls.append(args))

    config = {"bot": {"report_interval_hours": 1}}
    state_manager = _state_manager(tmp_path)

    with pytest.raises(StopLoop):
        await scheduled_report_loop(config, state_manager, initial_delay_seconds=5)

    assert sleeps == [5, 3600, 3600]
    assert build_calls == ["logs/live_trade_telemetry.csv", "logs/live_trade_telemetry.csv"]
    assert len(alert_calls) == 2


@pytest.mark.asyncio
async def test_scheduled_report_loop_swallows_errors(monkeypatch, tmp_path):
    class StopLoop(Exception):
        pass

    calls = {"count": 0}

    async def fake_sleep(seconds):
        calls["count"] += 1
        if calls["count"] >= 2:
            raise StopLoop

    def broken_build(path):
        raise RuntimeError("boom")

    monkeypatch.setattr("bot.weekly_report.asyncio.sleep", fake_sleep)
    monkeypatch.setattr("bot.weekly_report.build_weekly_report", broken_build)
    monkeypatch.setattr("bot.weekly_report.telemetry_path", lambda: "logs/live_trade_telemetry.csv")

    config = {"bot": {"report_interval_hours": 1}}
    state_manager = _state_manager(tmp_path)

    # A failure inside the loop body must not propagate out - only the
    # deliberate StopLoop from fake_sleep should ever escape.
    with pytest.raises(StopLoop):
        await scheduled_report_loop(config, state_manager, initial_delay_seconds=0)
