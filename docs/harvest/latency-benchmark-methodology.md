# Harvested — latency-benchmark methodology

For **G0** (generator bake-off on the A4000, due Aug 2) and for **Table 4** (cost/latency).
Source: `benchmark.py` @ `7c5b86f`. **Methodology only** — the Ollama HTTP transport is replaced by
vLLM, so none of the code carries over.

## The protocol that produced the 88 s/query figure

1. **Warm up before timing.** A tiny request (`prompt="Hi"`, `num_predict=5`) loads the model, then
   `sleep(3)` to let the runner settle. Cold-start time is a separate quantity and must not
   contaminate the per-call distribution.
2. **Cap output tokens** (`num_predict`) so latency reflects the model, not sampling luck.
3. **Vary the prompt across calls** — the base repo appended `(Call {i})` — so a prefix cache cannot
   silently serve call 2 onward. Under vLLM this matters *more*, not less: automatic prefix caching
   will otherwise make repeated benchmarks look free.
4. **Record wall-clock separately from the server's own timing.** The base repo captured
   `wall_time_s` alongside Ollama's `eval_count` / `eval_duration`, and derived tokens/sec from the
   server's numbers. The gap between the two is queueing and transport, and it is real cost.
5. **Sample peak memory on a background thread** during the call (0.2 s poll), not before or after.
6. **Report the range, never only the mean.** The base repo's headline was *~88 s, range 61–110 s* —
   a 1.8× spread that a mean alone would have hidden.
7. **Record system context in the same artifact** as the numbers (`get_system_info()`), so a result
   is never separable from the machine that produced it.

## Adaptation for G0

| Base repo | Here |
|---|---|
| Ollama `/api/generate`, `eval_count` / `eval_duration` | vLLM; take token counts from the API response usage fields |
| Peak **RSS** of the runner process | Peak **VRAM** (`torch.cuda.max_memory_allocated`, or `nvidia-smi` polling) — the constraint is 16 GB, and all three models must be co-resident (~9 GB) |
| 5 calls on a generic ML prompt | **10 real dev-split queries** with the actual citation-bearing prompt. G0 judges **citation-format compliance**, not general fluency — a model that is fast and ignores the `[1]` format is disqualified. |
| Reported to a log file | Written into `research_roadmap.md` §2 as the measured per-call latency, per G0 |

Everything above also applies to the **MedCPT encode-throughput measurement on 1,000 abstracts**
that G0 requires — warm up first, exclude model load, report the range, extrapolate to 2M only after.

## Carry the discipline forward

The one habit worth keeping past G0: the base repo's benchmark wrote its system info, its per-call
records, and its summary into a single artifact. That is the run-manifest idea, arrived at
independently — and it is why `harness.py` exists. When Table 4 is populated in W5, its per-query
`$`, input/output tokens, and wall-clock come from `costs.jsonl` written by the same protocol, not
from a separate benchmarking script run once in a good mood.
