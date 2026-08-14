#!/usr/bin/env python3
"""W4 smoke test: drive `generate_one` through the live vLLM server on the A4000, once per system.

This is **not** a gate run and computes no gate number. It answers a narrower question that has
never been answered against real hardware: does the generation path — prompt construction, the HTTP
boundary, response parsing, cost accounting, and `QueryRecord.validate()` — survive contact with an
actual 8B AWQ model? Every component has only ever seen an injected completer.

Three things it deliberately does:

1. **Fails before spending GPU time if the model id is wrong.** vLLM serves the model under the
   exact string passed to `--model` at launch, and `GenerationConfig.model` defaults to `""`
   because G0 chooses it. A typo would otherwise surface as a 404 per query, after N model loads.
   `/v1/models` is checked first and the served ids are printed on mismatch.

2. **Runs all three systems on the same questions.** Post-hoc is the two-call path and the only one
   where `raw_generation` carries a stage separator; joint and vanilla are one call. A smoke test
   that only exercised joint would leave the two-call orchestration unproven on real output.

3. **Reports `errors` and `violations` separately and repairs neither.** `errors` is "the grammar
   did not parse" (G2's valid-parse rate); `violations` is the ≤3-citation cap and "vanilla carries
   no citations". Both are measurements — see `Generation`'s docstring. A smoke run whose records
   were quietly cleaned would prove the wrong thing.

`--fake` swaps in a canned completer and touches no network, which is what the `Completer` seam in
`generate.py` exists for: the box is copy-paste only, so orchestration bugs are found here rather
than on the GPU. Fake runs stamp their `run_id` so their records can never be mistaken for real
ones.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from biomedqa.config import GenerationConfig  # noqa: E402
from biomedqa.generate import generate_one, split_stages  # noqa: E402
from biomedqa.prompts import CONTEXT_DEPTH, MAX_CLAIM_WORDS  # noqa: E402
from biomedqa.schema import (  # noqa: E402
    CostRecord,
    RetrievedPassage,
    System,
    to_dict,
    write_jsonl,
)

RUN_ID = "w4_generate_smoke"


def load_contexts(path: Path, n: int) -> list[dict]:
    """The first `n` questions from the committed top-10 dev contexts, in file order.

    File order, not a sample: this is a smoke test, and a seeded subsample would imply the result
    generalises to the dev set. It does not, and `--n 3` is not a measurement.
    """
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
            if len(out) == n:
                break
    if len(out) < n:
        raise SystemExit(f"{path} holds {len(out)} contexts, asked for {n}")
    return out


def passages_of(ctx: dict) -> list[RetrievedPassage]:
    return [
        RetrievedPassage(
            passage_id=p["passage_id"],
            rank=p["rank"],
            score=p["score"],
            retriever=p["retriever"],
            text=p["text"],
        )
        for p in ctx["passages"]
    ]


def assert_served(base_url: str, model: str, timeout: float) -> None:
    """Confirm the server is up and serving exactly `model`, before any generation is attempted."""
    try:
        with httpx.Client(base_url=base_url, timeout=timeout) as client:
            served = [m["id"] for m in client.get("/v1/models").json()["data"]]
    except Exception as exc:
        raise SystemExit(
            f"cannot reach a vLLM server at {base_url}: {exc}\n"
            "Start it per docs/harvest/runbooks/wsl-vllm-a4000.md §5, or tunnel to the box."
        ) from exc
    if model not in served:
        raise SystemExit(
            f"{base_url} does not serve {model!r}.\nServed: {served}\n"
            "Pass --model exactly as it appears in /v1/models."
        )
    print(f"vLLM at {base_url} serving {model}")


def _context_passages(prompt: str) -> list[tuple[str, str]]:
    """`(passage_id, text)` pairs recovered from the rendered context block.

    `render_context` emits `[passage_id]` on its own line followed by the passage text, so the
    canned completer can quote *verbatim* from what the prompt actually listed. Inventing a quote
    would fail `locate_quote` and exercise only the error branch — which is the opposite of what an
    offline orchestration check is for.
    """
    found = []
    lines = prompt.splitlines()
    for i, line in enumerate(lines[:-1]):
        m = re.fullmatch(r"\[([^\]\s]+)\]", line.strip())
        if m and lines[i + 1].strip():
            found.append((m.group(1), lines[i + 1]))
    return found


def _fake_completer(prompt: str, config: GenerationConfig, **kw) -> tuple[str, CostRecord]:
    """Canned output in the real line grammar. Exercises orchestration, parsing and the citation
    cap — never a substitute for a real run.

    Whether to cite is read off the prompt's own format block rather than off the system: vanilla's
    block omits `CITE` entirely, and `QueryRecord.validate()` requires vanilla to carry no
    citations. Keying on the prompt keeps the two in step without this script re-deriving the rule.
    """
    ctx = _context_passages(prompt)
    if len(ctx) < 2:
        raise AssertionError(f"canned completer found {len(ctx)} passages in the prompt, need 2")
    (pid1, text1), (pid2, text2) = ctx[0], ctx[1]
    sep = "||"
    # Bracketed ids, exactly as the format block now teaches and as render_context prints them.
    # This is also the offline exercise of the parser's bracket-stripping path.
    lines = ["DECISION: maybe", "CLAIM 1: Utilisation varies beyond what population need explains."]
    if "CITE:" in prompt:
        lines.append(f"CITE: [{pid1}] {sep} {text1[:60].strip()}")
        lines.append(f"CITE: [{pid2}] {sep} {text2[:60].strip()}")
    lines.append("CLAIM 2: The variation persists after case-mix adjustment.")
    if "CITE:" in prompt:
        lines.append(f"CITE: [{pid1}] {sep} {text1[:40].strip()}")
    return (
        "\n".join(lines),
        CostRecord(
            run_id=kw.get("run_id", ""),
            query_id=kw.get("query_id"),
            component="generate",
            backend=f"vllm:{config.model}",
            input_tokens=1024,
            output_tokens=32,
            usd=None,
            wall_s=0.0,
        ),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="model id exactly as served by /v1/models")
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--contexts", type=Path, default=Path("docs/harvest/dev_contexts_top10.jsonl"))
    ap.add_argument("--n", type=int, default=3, help="questions (default: %(default)s; a smoke test, not a sample)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--depth", type=int, default=CONTEXT_DEPTH)
    ap.add_argument("--max-tokens", type=int, default=GenerationConfig().max_tokens)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument(
        "--frequency-penalty",
        type=float,
        default=GenerationConfig().frequency_penalty,
        help=(
            "output-side repetition control, applied identically to all three systems. Sweep it to "
            "choose a value: read `over_length_claims` (should fall) against `quote_not_found` "
            "(must not rise) in the summary. Never repetition_penalty — vLLM applies that one over "
            "prompt tokens too, which penalises the verbatim quotes citations are made of."
        ),
    )
    ap.add_argument("--out-prefix", type=Path, default=Path("docs/harvest/generate_smoke"))
    ap.add_argument("--fake", action="store_true", help="canned completer; no network, no GPU")
    args = ap.parse_args()

    run_id = RUN_ID + ("_fakecompleter" if args.fake else "")
    completer = _fake_completer if args.fake else None

    if args.fake:
        print("--fake: canned completer, no network. Records are stamped and are not a real run.")
    else:
        # backends.complete reads this; set it before the first call rather than threading a URL
        # through generate_one, which has no business knowing about transport.
        os.environ["VLLM_BASE_URL"] = args.base_url
        assert_served(args.base_url, args.model, args.timeout)

    contexts = load_contexts(args.contexts, args.n)
    config = GenerationConfig(
        backend="vllm",
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=0.0,
        frequency_penalty=args.frequency_penalty,
    )

    records, costs, rows = [], [], []
    for ctx in contexts:
        passages = passages_of(ctx)
        for system in System:
            kwargs = {"complete": completer} if completer else {}
            gen = generate_one(
                ctx["question"],
                passages,
                ctx["gold_passage_ids"],
                system=system,
                config=config,
                seed=args.seed,
                run_id=run_id,
                query_id=ctx["query_id"],
                depth=args.depth,
                **kwargs,
            )
            rec = gen.record
            n_cit = sum(len(c.citations) for c in rec.claims)
            stages = len(split_stages(rec.raw_generation))
            records.append(rec)
            costs.extend(gen.costs)
            rows.append(
                {
                    "query_id": rec.query_id,
                    "system": system.value,
                    "stages": stages,
                    "claims": len(rec.claims),
                    "citations": n_cit,
                    "decision": rec.final_decision,
                    "errors": list(gen.errors),
                    "violations": list(gen.violations),
                    "latency_s": rec.latency_s,
                    "prompt_tokens": rec.prompt_tokens,
                    "completion_tokens": rec.completion_tokens,
                }
            )
            # The two counters the --frequency-penalty sweep is read off. They move in opposite
            # directions: the penalty is what stops a runaway claim, and too much of it pushes a
            # verbatim quote off its exact wording, which `locate_quote` then refuses. A value is
            # only acceptable if the first falls while the second does not rise.
            rows[-1]["over_length_claims"] = sum(
                1 for e in gen.errors if "max claim length" in e
            )
            rows[-1]["quote_not_found"] = sum(
                1 for e in gen.errors if "not found verbatim" in e
            )
            rows[-1]["longest_claim_words"] = max(
                (len(c.text.split()) for c in rec.claims), default=0
            )
            print(
                f"{rec.query_id:>9s} {system.value:8s} stages={stages} claims={len(rec.claims):2d} "
                f"cites={n_cit:2d} decision={str(rec.final_decision):5s} "
                f"errors={len(gen.errors)} violations={len(gen.violations)} "
                f"{rec.latency_s or 0.0:6.2f}s"
            )
            for problem in (*gen.errors, *gen.violations):
                print(f"{'':>9s}   ! {problem}")

    per_system = {}
    for system in System:
        mine = [r for r in rows if r["system"] == system.value]
        lat = [r["latency_s"] for r in mine if r["latency_s"] is not None]
        per_system[system.value] = {
            "n": len(mine),
            "clean_parses": sum(1 for r in mine if not r["errors"]),
            "records_with_violations": sum(1 for r in mine if r["violations"]),
            "stages_seen": sorted({r["stages"] for r in mine}),
            "mean_claims": round(statistics.fmean(r["claims"] for r in mine), 2),
            "total_citations": sum(r["citations"] for r in mine),
            "over_length_claims": sum(r["over_length_claims"] for r in mine),
            "quote_not_found": sum(r["quote_not_found"] for r in mine),
            "longest_claim_words": max((r["longest_claim_words"] for r in mine), default=0),
            "median_latency_s": round(statistics.median(lat), 2) if lat else None,
        }

    print("\nper system:")
    for name, s in per_system.items():
        print(
            f"  {name:8s} clean_parses {s['clean_parses']}/{s['n']}  "
            f"violations {s['records_with_violations']}/{s['n']}  "
            f"stages {s['stages_seen']}  mean_claims {s['mean_claims']}  "
            f"median {s['median_latency_s']}s"
        )

    expected = {System.JOINT.value: [1], System.POST_HOC.value: [2], System.VANILLA.value: [1]}
    stage_ok = all(per_system[k]["stages_seen"] == v for k, v in expected.items())
    print(
        f"\nstage-count check: {'PASSES' if stage_ok else 'FAILS'} "
        "(post_hoc must be two calls, joint and vanilla one)"
    )

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    rec_path = args.out_prefix.with_suffix(".records.jsonl")
    cost_path = args.out_prefix.with_suffix(".costs.jsonl")
    sum_path = args.out_prefix.with_suffix(".summary.json")
    write_jsonl(rec_path, records)
    write_jsonl(cost_path, costs)
    sum_path.write_text(
        json.dumps(
            {
                "script": "scripts/generate_smoke.py",
                "purpose": "W4 live-path smoke test; not a gate run and not a sample",
                "run_id": run_id,
                "fake_completer": args.fake,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "config": {
                    "model": args.model,
                    "base_url": None if args.fake else args.base_url,
                    "n_questions": args.n,
                    "seed": args.seed,
                    "depth": args.depth,
                    "max_tokens": args.max_tokens,
                    "temperature": 0.0,
                    "frequency_penalty": args.frequency_penalty,
                    "max_claim_words": MAX_CLAIM_WORDS,
                    "max_citations": config.max_citations,
                },
                "stage_count_check": {"expected": expected, "passed": stage_ok},
                "per_system": per_system,
                "rows": rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nWritten to {rec_path}, {cost_path}, {sum_path}")
    return 0 if stage_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
