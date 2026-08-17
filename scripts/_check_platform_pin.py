from vllm.platforms import current_platform

print("Platform:", current_platform)
print("is_pin_memory_available:", current_platform.is_pin_memory_available())
