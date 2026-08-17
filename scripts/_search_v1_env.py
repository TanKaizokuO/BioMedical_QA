import vllm.envs

with open(vllm.envs.__file__) as f:
    content = f.read()

for line in content.splitlines():
    if "v1" in line.lower():
        print(line)
