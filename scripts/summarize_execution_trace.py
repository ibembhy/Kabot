from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def _load_events(path: Path) -> list[dict]:
    events: list[dict] = []
    if not path.exists():
        return events
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def summarize(events: list[dict]) -> dict:
    starts = [event for event in events if event.get("event") == "execution_session_start"]
    completes = [event for event in events if event.get("event") == "execution_session_complete"]
    submitted = [event for event in events if event.get("event") == "execution_attempt_submitted"]
    skipped = [event for event in events if event.get("event") == "execution_attempt_skipped_no_depth"]

    by_market: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for event in events:
        key = (str(event.get("market_ticker") or ""), str(event.get("event") or ""))
        by_market[key].append(event)

    total_filled = sum(int(event.get("filled_contracts", 0) or 0) for event in completes)
    total_attempts = len(submitted)
    completed_sessions = len(completes)
    successful_sessions = sum(1 for event in completes if int(event.get("filled_contracts", 0) or 0) > 0)
    strategy_counter = Counter(str(event.get("strategy_name") or "unknown") for event in completes)
    status_counter = Counter(str(event.get("status") or "unknown") for event in completes)

    return {
        "sessions_started": len(starts),
        "sessions_completed": completed_sessions,
        "successful_sessions": successful_sessions,
        "session_fill_rate": round((successful_sessions / completed_sessions) * 100.0, 2) if completed_sessions else 0.0,
        "attempts_submitted": total_attempts,
        "attempts_skipped_no_depth": len(skipped),
        "total_filled_contracts": total_filled,
        "completed_statuses": dict(status_counter),
        "completed_strategies": dict(strategy_counter),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Kabot execution trace JSONL.")
    parser.add_argument("path", nargs="?", default="data/execution_trace.jsonl")
    args = parser.parse_args()
    path = Path(args.path)
    summary = summarize(_load_events(path))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
