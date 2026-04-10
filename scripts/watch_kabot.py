from __future__ import annotations

import re
import subprocess
import sys
import time


SERVICE_NAME = "kabot-live-kxbtc15m.service"
SAMPLE_PATTERN = re.compile(r"sample\[(.*)\]$")
ORDER_PATTERN = re.compile(r"order\[(.*)\]$")
PROFILE_LABELS = {
    "baseline_live": "Kabot",
    "exp_12m_signal_break": "K12+SE",
    "exp_12m_signal_break_execution": "+Fills",
    "exp_fills2": "+Fills2",
}


def _read_latest_line() -> str:
    result = subprocess.run(
        ["journalctl", "-u", SERVICE_NAME, "-n", "1", "--no-pager", "-o", "cat"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip()


def _parse_line(line: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    order_text = ""
    order_match = ORDER_PATTERN.search(line)
    if order_match:
        order_text = order_match.group(1)
        line = line[: order_match.start()].strip()

    sample_match = SAMPLE_PATTERN.search(line)
    sample_text = ""
    if sample_match:
        sample_text = sample_match.group(1)
        line = line[: sample_match.start()].strip()

    tokens = line.split()
    if tokens and "=" not in tokens[0] and "T" in tokens[0]:
        parsed["observed_at"] = tokens[0]
        tokens = tokens[1:]

    for token in tokens:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        parsed[key] = value

    if sample_text:
        for token in sample_text.split():
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            parsed[key] = value
    if order_text:
        for token in order_text.split():
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            parsed[f"order_{key}"] = value
    return parsed


def _short_time(value: str) -> str:
    if "T" not in value:
        return value or "-"
    try:
        return value.split("T", 1)[1][:8]
    except Exception:
        return value


def _format_row(parsed: dict[str, str], raw_line: str) -> str:
    if not raw_line:
        return "no log line yet"
    profile = parsed.get("profile", "-")
    row = [
        _short_time(parsed.get("observed_at", "-")),
        f"pf {PROFILE_LABELS.get(profile, profile)}",
        f"spot {parsed.get('spot', '-')}",
        f"vol {parsed.get('volume', '-')}",
        f"y/n {parsed.get('yes_ask_c', '-')}/{parsed.get('no_ask_c', '-')}",
        f"tte {parsed.get('tte_s', '-')}s",
        f"cand {parsed.get('candidates', '-')}",
        f"ord {parsed.get('orders', '-')}",
    ]
    order_side = parsed.get("order_side")
    if order_side:
        row.append(f"side {order_side}")
    order_strategy = parsed.get("order_strategy_name")
    if order_strategy:
        row.append(f"strat {order_strategy}")
    order_status = parsed.get("order_status")
    if order_status:
        row.append(f"ost {order_status}")
    order_signal = parsed.get("order_signal_price_cents")
    if order_signal:
        row.append(f"sig {order_signal}")
    order_exec = parsed.get("order_execution_price_cents")
    if order_exec:
        row.append(f"x {order_exec}")
    order_edge = parsed.get("order_gbm_edge")
    if order_edge:
        row.append(f"edge {order_edge}")
    order_depth = parsed.get("order_orderbook_available_contracts")
    if order_depth is not None:
        row.append(f"depth {order_depth}")
    order_fill = parsed.get("order_exchange_filled_contracts") or parsed.get("order_filled_contracts")
    if order_fill is not None:
        row.append(f"fill {order_fill}")
    rejects = parsed.get("rejects", "-")
    row.append(f"rej {rejects}")
    return " | ".join(row)


def main() -> None:
    interval = 2.0
    if len(sys.argv) > 1:
        try:
            interval = max(float(sys.argv[1]), 0.5)
        except ValueError:
            pass

    print("Kabot Live Watch")
    print("time | pf | spot | vol | y/n | tte | cand | ord | side | strat | ost | sig | x | edge | depth | fill | rej")
    print("-" * 140)
    last_raw_line = None
    while True:
        raw_line = _read_latest_line()
        if raw_line != last_raw_line:
            parsed = _parse_line(raw_line)
            try:
                print(_format_row(parsed, raw_line), flush=True)
            except BrokenPipeError:
                raise SystemExit(0)
            last_raw_line = raw_line
        time.sleep(interval)


if __name__ == "__main__":
    main()
