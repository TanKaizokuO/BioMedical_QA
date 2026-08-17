import inspect
from vllm.platforms import current_platform

print(inspect.getsource(current_platform.is_pin_memory_available))
