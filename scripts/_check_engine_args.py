from vllm.engine.arg_utils import AsyncEngineArgs

print([attr for attr in dir(AsyncEngineArgs) if "v1" in attr or "engine" in attr or "use" in attr])
