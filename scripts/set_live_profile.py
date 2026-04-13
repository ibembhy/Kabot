from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


SERVICE_PATH = Path("/etc/systemd/system/kabot-live-kxbtc15m.service")
BASE_COMMAND = (
    "/opt/kabot/venv/bin/kabot live-trade --series KXBTC15M --poll-seconds 2 "
    "--max-open-markets 3 --daily-loss-stop-dollars 10 --cooldown-loss-streak 3 "
    "--cooldown-minutes 30 --max-spot-age-seconds 30 --max-market-age-seconds 30"
)
PROFILE_COMMANDS = {
    "baseline_live": f"ExecStart={BASE_COMMAND}",
    "exp_12m_signal_break": f"ExecStart={BASE_COMMAND.replace('live-trade', 'live-trade --profile exp_12m_signal_break')}",
    "exp_12m_signal_break_execution": f"ExecStart={BASE_COMMAND.replace('live-trade', 'live-trade --profile exp_12m_signal_break_execution')}",
    "exp_fills2": f"ExecStart={BASE_COMMAND.replace('live-trade', 'live-trade --profile exp_fills2')}",
    "GOD": f"ExecStart={BASE_COMMAND.replace('live-trade', 'live-trade --profile GOD')}",
}


def set_profile(profile: str) -> None:
    if profile not in PROFILE_COMMANDS:
        raise SystemExit(f"Unknown profile: {profile}")
    text = SERVICE_PATH.read_text()
    lines = text.splitlines()
    updated = False
    for index, line in enumerate(lines):
        if line.startswith("ExecStart="):
            lines[index] = PROFILE_COMMANDS[profile]
            updated = True
            break
    if not updated:
        raise SystemExit("Could not find ExecStart in service file")
    SERVICE_PATH.write_text("\n".join(lines) + "\n")
    subprocess.run(["systemctl", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "restart", "kabot-live-kxbtc15m.service"], check=True)
    subprocess.run(["systemctl", "is-active", "kabot-live-kxbtc15m.service"], check=True)
    print(f"live profile set to {profile}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Switch Kabot live service between named profiles.")
    parser.add_argument("profile", choices=sorted(PROFILE_COMMANDS))
    args = parser.parse_args()
    set_profile(args.profile)


if __name__ == "__main__":
    main()
