from __future__ import annotations

import argparse
import csv
import re
import shlex
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path


DEFAULT_HOST = "root@66.135.8.30"
DEFAULT_SERVICE = "kabot-live-kxbtc15m.service"
DEFAULT_OUTPUT = Path("data/live_bankroll.csv")
SAMPLE_PATTERN = re.compile(r"sample\[(.*)\]$")
ORDER_PATTERN = re.compile(r"order\[(.*)\]$")
FIELDNAMES = [
    "recorded_at",
    "observed_at",
    "profile",
    "series",
    "status",
    "spot",
    "balance_usd",
    "bankroll_usd",
    "deployed_usd",
    "day_pnl_usd",
    "day_trades",
    "active_markets",
    "candidates",
    "orders",
    "signal_exits",
    "gbm_vol",
]


def _parse_line(line: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    order_match = ORDER_PATTERN.search(line)
    if order_match:
        line = line[: order_match.start()].strip()

    sample_match = SAMPLE_PATTERN.search(line)
    if sample_match:
        sample_text = sample_match.group(1)
        line = line[: sample_match.start()].strip()
        for token in sample_text.split():
            if "=" in token:
                key, value = token.split("=", 1)
                parsed[f"sample_{key}"] = value

    tokens = line.split()
    if tokens and "=" not in tokens[0] and "T" in tokens[0]:
        parsed["observed_at"] = tokens[0]
        tokens = tokens[1:]

    for token in tokens:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        parsed[key] = value
    return parsed


def _cents_to_usd(value: str | None) -> str:
    if value in (None, "", "na"):
        return ""
    try:
        return f"{int(value) / 100:.2f}"
    except ValueError:
        return ""


def _row_from_line(line: str) -> dict[str, str] | None:
    parsed = _parse_line(line)
    if "bankroll_cents" not in parsed and "balance_cents" not in parsed:
        return None
    return {
        "recorded_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "observed_at": parsed.get("observed_at", ""),
        "profile": parsed.get("profile", ""),
        "series": parsed.get("series", ""),
        "status": parsed.get("status", ""),
        "spot": parsed.get("spot", ""),
        "balance_usd": _cents_to_usd(parsed.get("balance_cents")),
        "bankroll_usd": _cents_to_usd(parsed.get("bankroll_cents")),
        "deployed_usd": _cents_to_usd(parsed.get("deployed_cents")),
        "day_pnl_usd": _cents_to_usd(parsed.get("day_pnl_cents")),
        "day_trades": parsed.get("day_trades", ""),
        "active_markets": parsed.get("active", ""),
        "candidates": parsed.get("candidates", ""),
        "orders": parsed.get("orders", ""),
        "signal_exits": parsed.get("signal_exits", ""),
        "gbm_vol": parsed.get("gbm_vol", ""),
    }


def _parse_observed_at(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _journal_command(args: argparse.Namespace) -> list[str]:
    journal_args = [
        "journalctl",
        "-u",
        args.service,
    ]
    if args.since:
        journal_args.extend(["--since", args.since])
    if args.until:
        journal_args.extend(["--until", args.until])
    journal_args.extend(
        [
        "-n",
        str(args.initial_lines),
        "--no-pager",
        "-o",
        "cat",
        ]
    )
    if not args.once:
        journal_args.append("-f")
    if args.host:
        return ["ssh", args.host, " ".join(shlex.quote(part) for part in journal_args)]
    return journal_args


def _open_writer(path: Path) -> tuple[csv.DictWriter, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    handle = path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
    if needs_header:
        writer.writeheader()
        handle.flush()
    return writer, handle


def main() -> int:
    parser = argparse.ArgumentParser(description="Tail Kabot live logs into a CSV for Excel/Sheets dashboards.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="SSH target. Use an empty string to read local journalctl.")
    parser.add_argument("--service", default=DEFAULT_SERVICE, help="systemd service to read")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="CSV file to append")
    parser.add_argument("--initial-lines", type=int, default=200, help="historical journal lines to seed the CSV")
    parser.add_argument("--since", help="journalctl --since value, for example '2026-04-06' or '2 hours ago'")
    parser.add_argument("--until", help="journalctl --until value, for example '2026-04-11 18:00:00'")
    parser.add_argument(
        "--sample-minutes",
        type=float,
        default=15.0,
        help="minimum minutes between rows written to the CSV; use 0 to keep every log row",
    )
    parser.add_argument("--once", action="store_true", help="write the matching historical rows and exit")
    args = parser.parse_args()

    output = Path(args.output)
    writer, handle = _open_writer(output)
    command = _journal_command(args)
    print(f"Writing bankroll rows to {output.resolve()}", flush=True)

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    assert process.stdout is not None
    rows_written = 0
    min_spacing = timedelta(minutes=max(args.sample_minutes, 0.0))
    last_observed_at: datetime | None = None
    try:
        for line in process.stdout:
            row = _row_from_line(line.strip())
            if row is None:
                continue
            observed_at = _parse_observed_at(row["observed_at"])
            if min_spacing.total_seconds() > 0 and observed_at is not None and last_observed_at is not None:
                if observed_at < last_observed_at + min_spacing:
                    continue
            writer.writerow(row)
            handle.flush()
            rows_written += 1
            if observed_at is not None:
                last_observed_at = observed_at
            print(
                f"{row['observed_at']} bankroll=${row['bankroll_usd'] or '?'} "
                f"balance=${row['balance_usd'] or '?'} day_pnl=${row['day_pnl_usd'] or '?'}",
                flush=True,
            )
    except KeyboardInterrupt:
        pass
    finally:
        process.terminate()
        handle.close()

    if rows_written == 0:
        stderr = process.stderr.read() if process.stderr is not None else ""
        if stderr:
            print(stderr.strip(), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
