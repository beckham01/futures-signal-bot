"""Persistent bot cooldown and last-signal state."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

LOGGER = logging.getLogger(__name__)


@dataclass
class BotState:
    cooldowns: dict[str, float]
    last_signal: dict | None
    signals_today: int
    last_scan_time: float
    users: dict[str, dict] = field(default_factory=dict)
    admins: dict[str, dict] = field(default_factory=dict)
    access_codes: dict[str, dict] = field(default_factory=dict)


class StateManager:
    def __init__(self, path: str = "state/bot_state.json", cooldown_hours: int = 4):
        self.path = Path(path)
        self.cooldown_seconds = cooldown_hours * 3600
        self.state = self.load()

    def load(self) -> BotState:
        if not self.path.exists():
            return BotState(cooldowns={}, last_signal=None, signals_today=0, last_scan_time=0.0)
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            return BotState(
                cooldowns=dict(payload.get("cooldowns", {})),
                last_signal=payload.get("last_signal"),
                signals_today=int(payload.get("signals_today", 0)),
                last_scan_time=float(payload.get("last_scan_time", 0.0)),
                users=dict(payload.get("users", {})),
                admins=dict(payload.get("admins", {})),
                access_codes=dict(payload.get("access_codes", {})),
            )
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            LOGGER.warning("Could not load bot state from %s: %s", self.path, exc)
            return BotState(cooldowns={}, last_signal=None, signals_today=0, last_scan_time=0.0)

    def save(self, state: BotState | None = None) -> None:
        if state is not None:
            self.state = state
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(asdict(self.state), indent=2), encoding="utf-8")
        except OSError as exc:
            LOGGER.warning("Could not save bot state to %s: %s", self.path, exc)

    def is_on_cooldown(self, symbol: str, direction: str) -> bool:
        key = f"{symbol}_{direction}"
        timestamp = self.state.cooldowns.get(key)
        return timestamp is not None and time.time() - timestamp < self.cooldown_seconds

    def set_cooldown(self, symbol: str, direction: str) -> None:
        self.state.cooldowns[f"{symbol}_{direction}"] = time.time()
        self.save()
