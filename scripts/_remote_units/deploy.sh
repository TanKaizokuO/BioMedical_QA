#!/usr/bin/env bash
# Move the C7 measurement off the pts/0 session that the wsl.exe interop relay owns and onto
# `systemd --user`, which a lingering user keeps alive across SSH disconnects and distro idling.
set -x
mkdir -p "$HOME/.config/systemd/user"
mv -f /home/user/vllm-8b.service /home/user/biomedqa-run.service "$HOME/.config/systemd/user/"
chmod +x /home/user/run_measure.sh

# Stop the session-bound run and the server it parented. Both come back under the units.
pkill -f run_all.sh || true
pkill -f decompose_smoke.py || true
pkill -f 'vllm serve' || true
sleep 5

loginctl enable-linger user
systemctl --user daemon-reload
systemctl --user enable --now vllm-8b.service
systemctl --user enable biomedqa-run.service
systemctl --user is-enabled vllm-8b.service biomedqa-run.service
