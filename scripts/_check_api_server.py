import inspect
from vllm.entrypoints.openai import api_server

print(inspect.getsource(api_server.build_async_engine_client_from_engine_args))
