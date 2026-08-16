#!/usr/bin/env python3
"""Stop WSL2 from tearing the distro down between SSH polls.

`loginctl enable-linger` and `systemd --user` keep a *Linux* session alive, but WSL2 shuts the
whole distro down when the last Windows-side `wsl.exe` client detaches — which is every time an
SSH channel closes. Measured 2026-08-16: `uptime -s` equalled the poll timestamp on consecutive
polls, and `biomedqa-run.service` restarted from zero each time, so a 3.3 h run made no progress
between them.

The fix is a Windows scheduled task holding one `wsl.exe` client open forever. `sleep infinity`
costs nothing and gives WSL a reason not to reclaim the VM. Local dev convenience only; reads
`.env.local` the same way `_remote.py` does.

Usage: uv run --with paramiko python scripts/_remote_keepalive.py [install|status|remove]
"""

from __future__ import annotations

import sys
from pathlib import Path

import paramiko

TASK = "KeepWSLAlive"
DISTRO = "Ubuntu-24.04"

env = {}
for line in Path(".env.local").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        env[k.strip()] = v.strip()

action = sys.argv[1] if len(sys.argv) > 1 else "install"
user, password = env["SSH_USER"], env["SSH_PASS"]

if action == "install":
    commands = [
        # /f so a re-run replaces rather than fails; /sc onstart + stored credentials so the task
        # survives a host reboot without anyone logging in.
        f'schtasks /create /tn {TASK} /f /sc onstart /ru "{user}" /rp "{password}" '
        f'/tr "wsl.exe -d {DISTRO} -u root -e /bin/sleep infinity"',
        f"schtasks /run /tn {TASK}",
        f"schtasks /query /tn {TASK} /fo LIST",
    ]
elif action == "status":
    commands = [f"schtasks /query /tn {TASK} /fo LIST"]
elif action == "remove":
    commands = [f"schtasks /end /tn {TASK}", f"schtasks /delete /tn {TASK} /f"]
else:
    raise SystemExit(f"unknown action {action!r}")

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(
    env["SSH_HOST"], username=user, password=password,
    look_for_keys=False, allow_agent=False, timeout=30,
)
failed = 0
for command in commands:
    _, stdout, stderr = client.exec_command(command, timeout=None, get_pty=False)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    code = stdout.channel.recv_exit_status()
    # The password is in the argv of the create call; never echo the command itself.
    print(f"--- exit={code} ---")
    print(out.strip())
    if err.strip():
        print(err.strip(), file=sys.stderr)
    failed |= code
client.close()
raise SystemExit(failed)
