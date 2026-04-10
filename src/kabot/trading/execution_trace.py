from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return {key: _jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return value


class ExecutionTraceWriter:
    def __init__(self, path: str | None) -> None:
        self.path = Path(path).expanduser() if path else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: dict[str, Any]) -> None:
        if self.path is None:
            return
        payload = json.dumps(_jsonable(event), separators=(",", ":"), sort_keys=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(payload + "\n")
