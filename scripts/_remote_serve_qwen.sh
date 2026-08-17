#!/usr/bin/env bash
export PATH="/home/user/venvs/vllm-server/bin:$PATH"
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
exec /home/user/venvs/vllm-server/bin/vllm serve Qwen/Qwen2.5-14B-Instruct-AWQ --max-model-len 14336 --gpu-memory-utilization 0.90 --host 0.0.0.0 > /tmp/vllm_qwen.log 2>&1
