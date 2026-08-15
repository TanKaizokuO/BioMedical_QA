"""vLLM (local 8B AWQ on the A4000) | Anthropic (Opus 5) — **Table 4**.

Key decisions preserved from the original stub:

- **vLLM is reached over HTTP, not imported.** It pins torch exactly, and adding it to this
  project's dependencies backtracks the resolver to vllm 0.2.5 and drags pydantic to 1.10.x
  workspace-wide (see the note in ``pyproject.toml``). It runs on the box in its own environment;
  the generator backend is a network boundary. ``scripts/g0_generator_bakeoff.py`` is the working
  reference that proved this protocol.

- **Only the local backend is seedable.** The Claude API rejects ``temperature``/``top_p``/``top_k``
  with a 400, so the ≥3-seed plan is implementable locally and nowhere else (ADR-0004). This is
  why development iteration is local and the frontier model is the judge, not the generator.

- **Every call emits a** ``CostRecord``. Table 4's columns were fixed in W0 so that
  instrumentation happens here rather than being discovered missing in October.
"""

from __future__ import annotations

import os
import time

import httpx

from .config import GenerationConfig
from .schema import CostRecord

# Base URL for the vLLM OpenAI-compatible server.
# Override with VLLM_BASE_URL when the server is not on localhost:8000.
_VLLM_BASE_URL_DEFAULT = "http://localhost:8000"

# Timeout for vLLM requests — 8B AWQ generation can take a while.
_VLLM_TIMEOUT = 120.0

# Claude pricing (USD per million tokens) — snapshot 2025-07.
# Update here when Anthropic changes rates; the costs.jsonl audit trail makes drift visible.
_ANTHROPIC_PRICE: dict[str, tuple[float, float]] = {
    # model-prefix → (input $/M, output $/M)
    "claude-opus-5":         (15.00, 75.00),
    "claude-opus-4":         (15.00, 75.00),
    "claude-sonnet-4":        (3.00, 15.00),
    "claude-sonnet-3-7":      (3.00, 15.00),
    "claude-sonnet-3-5":      (3.00, 15.00),
    "claude-haiku-3-5":       (0.80,  4.00),
    "claude-haiku-3":         (0.25,  1.25),
}


def _anthropic_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Estimate USD cost using prefix-matched pricing table. Returns None on unknown model."""
    for prefix, (in_rate, out_rate) in _ANTHROPIC_PRICE.items():
        if model.startswith(prefix):
            return (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000
    return None


def complete(
    prompt: str,
    config: GenerationConfig,
    *,
    seed: int = 0,
    run_id: str = "",
    query_id: str | None = None,
) -> tuple[str, CostRecord]:
    """One completion plus its cost row. Backend is chosen by ``config.backend``.

    Args:
        prompt:   The fully-rendered prompt string.
        config:   Generation knobs — model, max_tokens, temperature, backend.
        seed:     RNG seed passed to vLLM (ignored for Anthropic — ADR-0004).
        run_id:   Propagated into ``CostRecord`` for cross-table joins.
        query_id: Propagated into ``CostRecord``; ``None`` for batch/offline calls.

    Returns:
        ``(text, cost_record)`` — the raw model output and its accounting row.

    Raises:
        ValueError: Unknown ``config.backend``.
        httpx.ConnectError: vLLM server unreachable (see runbook in README §G0).
    """
    if config.backend == "vllm":
        return _vllm_complete(prompt, config, seed=seed, run_id=run_id, query_id=query_id)
    if config.backend == "anthropic":
        return _anthropic_complete(prompt, config, seed=seed, run_id=run_id, query_id=query_id)
    raise ValueError(
        f"Unknown backend {config.backend!r}. Expected 'vllm' or 'anthropic'."
    )


def _vllm_complete(
    prompt: str,
    config: GenerationConfig,
    *,
    seed: int,
    run_id: str,
    query_id: str | None,
) -> tuple[str, CostRecord]:
    """POST to the vLLM OpenAI-compatible chat endpoint and build a CostRecord.

    Uses the chat-completions endpoint (``/v1/chat/completions``) rather than the raw
    completions endpoint — the bakeoff script (``g0_generator_bakeoff.py``) proved this
    is what the vLLM server exposes, and switching would change token counts.

    Connection errors bubble as ``httpx.ConnectError`` with a message that points the
    operator to the runbook: make sure vLLM is running on the A4000 and the port is
    forwarded before calling this.
    """
    base_url = os.environ.get("VLLM_BASE_URL", _VLLM_BASE_URL_DEFAULT).rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3].rstrip("/")

    # `frequency_penalty` and `stop` are OpenAI-standard top-level fields on vLLM's
    # /v1/chat/completions (`ChatCompletionRequest`), so they need no extra_body. `repetition_penalty`
    # is deliberately absent: vLLM applies it over prompt *and* output tokens, which would penalise
    # the verbatim quotes citations are made of — see `GenerationConfig.frequency_penalty`.
    body = {
        "model": config.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "frequency_penalty": config.frequency_penalty,
        "stop": list(config.stop),
        "seed": seed,
    }

    t0 = time.perf_counter()
    try:
        with httpx.Client(base_url=base_url, timeout=_VLLM_TIMEOUT) as client:
            resp = client.post("/v1/chat/completions", json=body)
    except httpx.ConnectError as exc:
        raise httpx.ConnectError(
            f"Cannot reach vLLM at {base_url}. "
            "Ensure the server is running on the A4000 and the port is forwarded. "
            f"Original error: {exc}"
        ) from exc
    wall_s = time.perf_counter() - t0

    resp.raise_for_status()
    data = resp.json()

    text: str = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    input_tokens: int | None = usage.get("prompt_tokens")
    output_tokens: int | None = usage.get("completion_tokens")

    cost = CostRecord(
        run_id=run_id,
        query_id=query_id,
        component="generate",
        backend=f"vllm:{config.model}",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        usd=None,          # local inference — no per-token USD cost
        wall_s=wall_s,
    )
    return text, cost


def _anthropic_complete(
    prompt: str,
    config: GenerationConfig,
    *,
    seed: int,  # noqa: ARG001 — Anthropic API has no seed parameter (ADR-0004)
    run_id: str,
    query_id: str | None,
) -> tuple[str, CostRecord]:
    """Call the Anthropic Messages API and build a CostRecord.

    Anthropic rejects ``temperature``, ``top_p``, and ``top_k`` with a 400 when the
    model is in extended-thinking mode, and they are not meaningful for reproducibility
    anyway (ADR-0004 — only the local backend is seedable). Neither is passed here.

    ``frequency_penalty`` has no Anthropic equivalent and is **refused** rather than dropped: a knob
    silently ignored on one backend is a run whose manifest claims a decoding setting the tokens
    never saw. ``stop`` does have an equivalent and is forwarded as ``stop_sequences``.

    ``seed`` is accepted in the signature for a uniform call-site interface but is
    silently ignored — Anthropic's API offers no seeding guarantee.
    """
    if config.frequency_penalty:
        raise ValueError(
            "frequency_penalty has no Anthropic equivalent; it would be dropped silently while the "
            f"manifest recorded {config.frequency_penalty}. Set it to 0.0 for backend='anthropic'."
        )

    import anthropic  # local import — not all environments install the SDK

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    t0 = time.perf_counter()
    message = client.messages.create(
        model=config.model,
        max_tokens=config.max_tokens,
        # temperature / top_p / top_k deliberately omitted — Anthropic returns 400 (ADR-0004)
        messages=[{"role": "user", "content": prompt}],
        **({"stop_sequences": list(config.stop)} if config.stop else {}),
    )
    wall_s = time.perf_counter() - t0

    text: str = message.content[0].text

    input_tokens: int | None = getattr(message.usage, "input_tokens", None)
    output_tokens: int | None = getattr(message.usage, "output_tokens", None)

    usd: float | None = None
    if input_tokens is not None and output_tokens is not None:
        usd = _anthropic_usd(config.model, input_tokens, output_tokens)

    cost = CostRecord(
        run_id=run_id,
        query_id=query_id,
        component="generate",
        backend=f"anthropic:{config.model}",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        usd=usd,
        wall_s=wall_s,
    )
    return text, cost
