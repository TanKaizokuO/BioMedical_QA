import subprocess

out = subprocess.check_output(
    ["/home/user/venvs/vllm-server/bin/vllm", "serve", "--help=all"]
).decode()
for line in out.splitlines():
    if any(k in line.lower() for k in ["eager", "v1", "engine", "uva", "utilization"]):
        print(line)
