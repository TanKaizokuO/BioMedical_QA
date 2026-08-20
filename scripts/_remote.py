#!/usr/bin/env python3
"""Throwaway remote-exec helper: run a command on the A4000 box over password SSH.

Local dev convenience only (the workstation has no GPU and no sshpass); reads .env.local.
Usage: uv run --with paramiko python scripts/_remote.py '<command>'
"""

from __future__ import annotations

import sys
from pathlib import Path

import paramiko

env = {}
for line in Path(".env.local").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        env[k.strip()] = v.strip()

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(
    env["SSH_HOST"], username=env["SSH_USER"], password=env["SSH_PASS"],
    look_for_keys=False, allow_agent=False, timeout=30,
)
if sys.argv[1] == "--put":
    # Local file -> WSL guest path, staged through the Windows-side temp dir: the SSH server is
    # Windows OpenSSH, so SFTP cannot see the guest filesystem directly.
    local, remote = sys.argv[2], sys.argv[3]
    staged = f"C:/Users/{env['SSH_USER']}/AppData/Local/Temp/_omp_put"
    sftp = client.open_sftp()
    sftp.put(local, staged)
    sftp.close()
    cmd = f"""wsl.exe -d Ubuntu-24.04 -- bash -lc 'cp "$(wslpath "{staged}")" "{remote}"'"""
elif sys.argv[1] == "--get":
    # WSL guest path -> Local file, staged through Windows temp dir
    remote, local = sys.argv[2], sys.argv[3]
    staged = f"C:/Users/{env['SSH_USER']}/AppData/Local/Temp/_omp_get"
    cmd = f"""wsl.exe -d Ubuntu-24.04 -- bash -lc 'cp "{remote}" "$(wslpath "{staged}")"'"""
    stdin, stdout, stderr = client.exec_command(cmd, timeout=None, get_pty=False)
    code = stdout.channel.recv_exit_status()
    if code != 0:
        err = stderr.read().decode("utf-8", "replace")
        sys.stderr.write(err)
        client.close()
        raise SystemExit(code)
    sftp = client.open_sftp()
    sftp.get(staged, local)
    sftp.close()
    client.close()
    raise SystemExit(0)
else:
    cmd = sys.argv[1]
stdin, stdout, stderr = client.exec_command(cmd, timeout=None, get_pty=False)
out = stdout.read().decode("utf-8", "replace")
err = stderr.read().decode("utf-8", "replace")
code = stdout.channel.recv_exit_status()
sys.stdout.write(out)
if err:
    sys.stderr.write(err)
client.close()
raise SystemExit(code)
