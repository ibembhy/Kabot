"""Check how kabot is running on server."""
import subprocess

HOST = "root@66.135.8.30"
SSH_BASE = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10"]

def ssh(cmd, timeout=20):
    try:
        r = subprocess.run(SSH_BASE + [HOST, cmd], capture_output=True, text=True, timeout=timeout)
        return (r.stdout + r.stderr).strip()
    except Exception as e:
        return str(e)

print("=== All running kabot processes ===")
print(ssh("ps aux | grep -i kabot | grep -v grep"))

print("\n=== All systemd services ===")
print(ssh("systemctl list-units --type=service --state=running 2>/dev/null | head -30"))

print("\n=== Recent kabot-related logs (last 60 lines) ===")
print(ssh("journalctl -n 60 --no-pager 2>/dev/null | grep -i kabot || echo 'no journal entries for kabot'"))

print("\n=== All log files ===")
print(ssh("find /root /opt /var/log -name '*.log' -newer /tmp -type f 2>/dev/null | head -20"))

print("\n=== Any nohup.out ===")
print(ssh("cat /root/nohup.out 2>/dev/null | tail -30 || echo 'no nohup.out'"))

print("\n=== Screen sessions ===")
print(ssh("screen -ls 2>/dev/null || echo 'no screen'"))

print("\n=== tmux sessions ===")
print(ssh("tmux ls 2>/dev/null || echo 'no tmux'"))

print("\n=== crontab ===")
print(ssh("crontab -l 2>/dev/null || echo 'no crontab'"))
