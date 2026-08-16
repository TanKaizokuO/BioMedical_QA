#!/bin/bash
# Read-only view of the C7 measurement. Everything now runs under `systemd --user`, so the unit
# state is the first thing worth seeing: a run that is gone shows up here as `inactive (dead)`
# rather than as a silently missing process.
echo "=== $(date -u +%H:%M:%S) units ==="
systemctl --user --no-pager --no-legend list-units 'vllm-8b.service' 'biomedqa-run.service' 2>&1
systemctl --user show biomedqa-run.service -p ActiveState -p SubState -p Result -p ExecMainStartTimestamp -p NRestarts 2>&1
echo "=== run_all.log ==="
tail -25 /home/user/run_all.log 2>/dev/null || echo "no run_all.log"
echo "=== procs (TT must be ? — a tty here means it is session-bound again) ==="
ps -eo pid,ppid,tty,etime,cmd | grep -E 'vllm serve|decompose_smoke|run_measure' | grep -v grep
echo "=== gpu ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo "=== sanity.log tail ==="
tail -8 /home/user/sanity.log 2>/dev/null
echo "=== guided_v2.log tail ==="
tail -5 /home/user/guided_v2.log 2>/dev/null
echo "=== progress (queries done of 100) ==="
grep -c "decompose_errors=" /home/user/guided_v2.log 2>/dev/null
echo "=== artifacts ==="
ls -la /home/user/BioMedical_QA/docs/harvest/_sanity_guided.summary.json \
       /home/user/BioMedical_QA/docs/harvest/decompose_guided_v2.* 2>/dev/null
