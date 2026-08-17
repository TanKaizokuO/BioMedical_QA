with open("/tmp/vllm_qwen.log") as f:
    text = f.read()

for line in text.splitlines():
    if any(k in line for k in ["ERROR", "Traceback", "OutOfMemoryError", "OOM", "CUDA out of memory"]):
        print(line)
