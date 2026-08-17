import inspect
from vllm.utils.platform_utils import is_pin_memory_available

print(inspect.getsource(is_pin_memory_available))
