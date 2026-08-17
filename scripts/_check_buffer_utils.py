with open("/home/user/venvs/vllm-server/lib/python3.12/site-packages/vllm/v1/worker/gpu/buffer_utils.py") as f:
    lines = f.readlines()
for i, line in enumerate(lines[:60]):
    print(f"{i+1}: {line}", end="")
