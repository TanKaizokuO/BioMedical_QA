#!/usr/bin/env bash
set -euo pipefail
echo "Starting Qwen2.5-14B-Instruct-AWQ download..."
/home/user/venvs/vllm-server/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-14B-Instruct-AWQ')"
echo "Download finished successfully."
