from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

try:
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


def _deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _coerce_env_value(raw: str, current: Any) -> Any:
    if isinstance(current, bool):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(current, int) and not isinstance(current, bool):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, list):
        return [item.strip() for item in raw.split(",") if item.strip()]
    return raw


@dataclass(frozen=True)
class Settings:
    raw: dict[str, Any]

    @property
    def app(self) -> dict[str, Any]:
        return self.raw["app"]

    @property
    def storage(self) -> dict[str, Any]:
        return self.raw["storage"]

    @property
    def data(self) -> dict[str, Any]:
        return self.raw["data"]

    @property
    def markets(self) -> dict[str, Any]:
        return self.raw["markets"]

    @property
    def model(self) -> dict[str, Any]:
        return self.raw["model"]

    @property
    def signal(self) -> dict[str, Any]:
        return self.raw["signal"]

    @property
    def trading(self) -> dict[str, Any]:
        return self.raw["trading"]

    @property
    def backtest(self) -> dict[str, Any]:
        return self.raw["backtest"]


def load_settings(config_path: str | Path | None = None) -> Settings:
    load_dotenv()
    root = Path(__file__).resolve().parents[2]
    default_path = root / "config" / "default.toml"
    with default_path.open("rb") as handle:
        config = tomllib.load(handle)

    effective_path = Path(config_path or os.getenv("KABOT_CONFIG_PATH", "")).expanduser() if (config_path or os.getenv("KABOT_CONFIG_PATH")) else None
    if effective_path:
        with effective_path.open("rb") as handle:
            config = _deep_merge(config, tomllib.load(handle))

    for section_name, section in config.items():
        if not isinstance(section, dict):
            continue
        for key, current_value in list(section.items()):
            env_name = f"KABOT_{section_name}_{key}".upper()
            raw_value = os.getenv(env_name)
            if raw_value:
                section[key] = _coerce_env_value(raw_value, current_value)

    return Settings(raw=config)

