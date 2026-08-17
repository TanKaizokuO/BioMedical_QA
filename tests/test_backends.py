"""What actually reaches the server. A knob recorded in the manifest but absent from the request
body is worse than no knob: the run claims a decoding setting the tokens never saw."""

from __future__ import annotations

import pytest

from biomedqa import backends
from biomedqa.config import GenerationConfig


class _FakeResponse:
    def __init__(self, text: str, status_code: int = 200, body: str = ""):
        self._text = text
        self.status_code = status_code
        self.text = body
        self.request = None

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {
            "choices": [{"message": {"content": self._text}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }


class _FakeClient:
    """Captures the POST body. Instantiated by `_vllm_complete` as a context manager."""

    last_body: dict | None = None

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, path: str, json: dict) -> _FakeResponse:
        type(self).last_body = json
        return _FakeResponse("DECISION: yes\nCLAIM 1: X.\n")


@pytest.fixture
def body(monkeypatch):
    """Return the request body produced for a config, via a captured fake client."""

    def _call(config: GenerationConfig) -> dict:
        monkeypatch.setattr(backends.httpx, "Client", _FakeClient)
        _FakeClient.last_body = None
        backends.complete("prompt", config, seed=0, run_id="r", query_id="q")
        assert _FakeClient.last_body is not None
        return _FakeClient.last_body

    return _call


def _config(**kw) -> GenerationConfig:
    return GenerationConfig(backend="vllm", model="m", **kw)


def test_the_frequency_penalty_reaches_the_request(body):
    """The whole point of CONFIG_VERSION 1.4.0. A value that stays in the dataclass leaves the
    731-word repetition loop in `parity_iter1b` unaddressed while the config says otherwise."""
    assert body(_config(frequency_penalty=0.3))["frequency_penalty"] == 0.3


def test_repetition_penalty_is_never_sent(body):
    """Verified in vLLM source (`model_executor/layers/utils.py::apply_penalties`): the penalty is
    applied over `prompt_mask | output_mask`, so it down-weights every token appearing in the
    prompt — exactly the tokens a citation must copy verbatim for `locate_quote` to find its span.
    Reaching for it as the obvious anti-repetition knob would trade a decoding defect for a
    citation defect, and Table 2 would read the loss as failed grounding. This test is the note
    that stops it being added."""
    sent = body(_config(frequency_penalty=0.3))

    assert "repetition_penalty" not in sent
    assert "presence_penalty" not in sent


def test_stop_sequences_reach_the_request(body):
    assert body(_config(stop=("\nQUESTION:",)))["stop"] == ["\nQUESTION:"]


def test_the_default_config_sends_the_chosen_frequency_penalty_and_no_stop(body):
    """The value was chosen on the A4000 (`docs/harvest/generate_fp_sweep.md`). What this test
    now defends is that the chosen value reaches the request rather than sitting in the dataclass.
    Sent explicitly rather than omitted: the body is a faithful image of the config, which is
    what makes a replayed request comparable to the run that produced it."""
    sent = body(_config())

    assert sent["frequency_penalty"] == 0.5
    assert sent["stop"] == []


def test_the_seedable_knobs_still_travel_together(body):
    """ADR-0004: the local backend is the only seedable one, so seed and temperature must both
    survive the payload change that added the penalties."""
    sent = body(_config(temperature=0.0, max_tokens=1536))

    assert sent["seed"] == 0
    assert sent["temperature"] == 0.0
    assert sent["max_tokens"] == 1536
def test_response_format_reaches_the_request(monkeypatch):
    """Slice A: response_format is forwarded to vLLM's structured output engine."""
    monkeypatch.setattr(backends.httpx, "Client", _FakeClient)
    _FakeClient.last_body = None
    rf = {"type": "json_schema", "json_schema": {"name": "test", "schema": {}}}
    backends.complete("prompt", _config(), seed=0, run_id="r", query_id="q", response_format=rf)
    assert _FakeClient.last_body is not None
    assert _FakeClient.last_body.get("response_format") == rf


def test_anthropic_response_format_uses_tool_choice(monkeypatch):
    """Anthropic backend converts json_schema response_format to tool_choice."""
    import sys
    class _FakeToolUse:
        type = "tool_use"
        input = {"claims": []}
    class _FakeMessage:
        content = [_FakeToolUse()]
        usage = type("Usage", (), {"input_tokens": 10, "output_tokens": 5})()
    class _FakeMessages:
        last_kw = None
        def create(self, **kw):
            _FakeMessages.last_kw = kw
            return _FakeMessage()
    class _FakeAnthropicClient:
        messages = _FakeMessages()

    fake_mod = type("FakeAnthropic", (), {"Anthropic": _FakeAnthropicClient})()
    monkeypatch.setitem(sys.modules, "anthropic", fake_mod)
    config = GenerationConfig(backend="anthropic", model="claude-opus-5", frequency_penalty=0.0)
    rf = {"type": "json_schema", "json_schema": {"name": "recitation", "schema": {"properties": {}}}}
    text, cost = backends.complete("prompt", config, seed=0, run_id="r", query_id="q", response_format=rf)

    assert text == '{"claims": []}'
    assert _FakeMessages.last_kw is not None
    assert "tools" in _FakeMessages.last_kw
    assert _FakeMessages.last_kw["tool_choice"] == {"type": "tool", "name": "recitation"}
def test_a_rejected_request_carries_the_servers_reason(monkeypatch):
    """A 400 that hides its body is a 45-minute diagnosis.

    The n=100 guided run died on 2026-08-16 with nothing but `Client error '400 Bad Request'` and
    a URL. The reason was in the response body all along — the prompt was 4327 tokens against a
    server with `--max-model-len 8192` and a 4096-token completion request. `raise_for_status()`
    drops that body, so the backend builds the error itself.
    """

    class _RejectingClient(_FakeClient):
        def post(self, path: str, json: dict) -> _FakeResponse:
            return _FakeResponse(
                "",
                status_code=400,
                body='{"error":{"message":"This model\'s maximum context length is 8192 tokens."}}',
            )

    monkeypatch.setattr(backends.httpx, "Client", _RejectingClient)
    with pytest.raises(backends.httpx.HTTPStatusError, match="maximum context length is 8192"):
        backends.complete("prompt", _config(), seed=0, run_id="r", query_id="q")


def test_anthropic_refuses_a_penalty_it_cannot_apply():
    """Anthropic has no frequency_penalty. Dropping it silently would produce a run whose manifest
    records a decoding setting that never reached the sampler — and because the two backends are
    compared in Table 4, the divergence would be invisible."""
    with pytest.raises(ValueError, match="no Anthropic equivalent"):
        backends.complete(
            "prompt",
            GenerationConfig(backend="anthropic", model="claude-opus-5", frequency_penalty=0.5),
            seed=0,
            run_id="r",
            query_id="q",
        )


def test_prompt_window_guard_raises_when_oversized(monkeypatch):
    """Prompt window guard verifies prompt_tokens + max_tokens <= model_max_len before request."""
    monkeypatch.setattr(backends.httpx, "Client", _FakeClient)
    # Create an oversized prompt exceeding model_max_len (8192 for model="m")
    oversized_prompt = "word " * 7000
    cfg = GenerationConfig(backend="vllm", model="m", max_tokens=2000)
    with pytest.raises(ValueError) as exc_info:
        backends.complete(oversized_prompt, cfg, seed=0, run_id="r", query_id="q")

    msg = str(exc_info.value)
    assert "Prompt window exceeded" in msg
    assert "prompt_tokens" in msg
    assert "max_tokens" in msg
    assert "model_max_len" in msg


def test_prompt_window_guard_passes_through_unchanged(monkeypatch):
    """When prompt_tokens + max_tokens <= model_max_len, request completes normally."""
    monkeypatch.setattr(backends.httpx, "Client", _FakeClient)
    _FakeClient.last_body = None
    normal_prompt = "Short prompt"
    cfg = GenerationConfig(backend="vllm", model="m", max_tokens=1536)
    text, cost = backends.complete(normal_prompt, cfg, seed=0, run_id="r", query_id="q")
    assert text.startswith("DECISION:")
    assert cost.run_id == "r"

