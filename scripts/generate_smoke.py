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

4. **Manifests the run before making it.** `<prefix>.manifest.json` is written by
   `harness.run_manifest` before the first record and finalized after the last, so every number this
   script produces can be traced to a config hash, an index fingerprint, a split hash and a commit —
   G5's condition. The knobs are no longer restated in the summary; `verify_run`'s findings are
   printed at the end, and a dirty tree is one of them.

`--fake` swaps in a canned completer and touches no network, which is what the `Completer` seam in
`generate.py` exists for: the box is copy-paste only, so orchestration bugs are found here rather
than on the GPU. Fake runs carry `_fakecompleter` in the **prefix** as well as the `run_id`, so
their records can never be mistaken for real ones and `run_id == prefix.name` still holds — which is
what ties a manifest to the records beside it.
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

from biomedqa.config import GenerationConfig, RunConfig  # noqa: E402
from biomedqa.generate import generate_one, split_stages  # noqa: E402
from biomedqa.harness import (  # noqa: E402
    costs_path,
    finalize_run,
    manifest_path,
    records_path,
    run_manifest,
    verify_run,
)
from biomedqa.prompts import CONTEXT_DEPTH, MAX_CLAIM_WORDS  # noqa: E402
from biomedqa.schema import (  # noqa: E402
    CostRecord,
    RetrievedPassage,
    System,
    to_dict,
    write_jsonl,
)


def load_contexts(
    path: Path, n: int = 3, query_ids: Sequence[str] | None = None
) -> list[dict]:
    """The dev contexts from `path`, in file order.

    If `query_ids` is specified, selects exactly those query IDs in file order.
    Otherwise, returns the first `n` contexts in file order.
    """
    out = []
    if query_ids:
        requested = list(query_ids)
        wanted = set(requested)
        found = set()
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                ctx = json.loads(line)
                qid = str(ctx.get("query_id"))
                if qid in wanted:
                    out.append(ctx)
                    found.add(qid)
                    if len(found) == len(wanted):
                        break
        missing = [qid for qid in requested if qid not in found]
        if missing:
            raise SystemExit(f"query ids not found in {path}: {', '.join(missing)}")
        return out

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
    norm_url = base_url.rstrip("/")
    if norm_url.endswith("/v1"):
        norm_url = norm_url[:-3].rstrip("/")
    try:
        with httpx.Client(base_url=norm_url, timeout=timeout) as client:
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
    ap.add_argument(
        "--query-ids",
        type=str,
        default=None,
        help=(
            "comma-separated list of query ids selecting exactly those contexts from --contexts, "
            "in file order. Target individual records carrying runaway pathology (indices 1, 25, 48, 99) "
            "so a targeted sweep point costs 12 questions instead of 100. Mutually exclusive with a non-default --n."
        ),
    )
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
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "delete this prefix's manifest, records and costs first. Without it a prefix that "
            "already holds a manifest is refused, because a second run under one manifest inherits "
            "the first run's provenance."
        ),
    )
    args = ap.parse_args()

    completer = _fake_completer if args.fake else None
    # The fake marker lives in the *prefix*, not only in the run id, so `run_id == prefix.name`
    # holds and the manifest can never describe records stamped with a different id. Fake records
    # remain unmistakable for real ones — now their filenames say so too.
    prefix = args.out_prefix
    if args.fake and not prefix.name.endswith("_fakecompleter"):
        prefix = prefix.with_name(prefix.name + "_fakecompleter")
    run_id = prefix.name

    if args.fake:
        print("--fake: canned completer, no network. Records are stamped and are not a real run.")
    else:
        # backends.complete reads this; set it before the first call rather than threading a URL
        # through generate_one, which has no business knowing about transport.
        os.environ["VLLM_BASE_URL"] = args.base_url
        assert_served(args.base_url, args.model, args.timeout)

    if args.query_ids is not None:
        if args.n != 3 or any(arg == "--n" or arg.startswith("--n=") for arg in sys.argv[1:]):
            raise SystemExit(
                f"cannot combine --query-ids and --n (passed --query-ids with --n {args.n}); choose one"
            )
        query_ids = [q.strip() for q in args.query_ids.split(",") if q.strip()]
    else:
        query_ids = None

    contexts = load_contexts(args.contexts, n=args.n, query_ids=query_ids)
    # A whole RunConfig, not a bare GenerationConfig: the manifest's job is to name every knob the
    # numbers rest on, and `--depth` is a retrieval knob that the summary used to record as a loose
    # integer next to knobs from a different section.
    run_config = RunConfig().ablate(
        run_id,
        **{
            "generation.backend": "vllm",
            "generation.model": args.model,
            "generation.max_tokens": args.max_tokens,
            "generation.frequency_penalty": args.frequency_penalty,
            "generation.seeds": (args.seed,),
            "retrieval.top_k": args.depth,
        },
    )
    config = run_config.generation

    if args.overwrite:
        for path in (manifest_path(prefix), records_path(prefix), costs_path(prefix)):
            path.unlink(missing_ok=True)
    # After the preflight, before the first record. Written at the end it would carry the tree and
    # the clock as they were once the run finished, and a crashed run would read as a complete one.
    try:
        manifest = run_manifest(run_config, prefix)
    except FileExistsError as exc:
        raise SystemExit(f"{exc} Pass --overwrite to do exactly that.")
    print(f"manifest: {manifest_path(prefix)}  config {manifest['config_hash']}  "
          f"git {manifest['git_sha']}")

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
                    "recovered": list(gen.recovered),
                    "latency_s": rec.latency_s,
                    "prompt_tokens": rec.prompt_tokens,
                    "completion_tokens": rec.completion_tokens,
                }
            )
            # The sweep counters the --frequency-penalty sweep is read off. They move in opposing
            # directions: frequency_penalty is what stops non-terminating generation loops (measured
            # by runaway_chain_claims and over_length_claims), but too much of it pushes a verbatim
            # quote off its exact wording, which `locate_quote` then refuses (`quote_not_found`). A
            # frequency_penalty value is acceptable only if runaway_chain_claims and
            # over_length_claims fall while quote_not_found does not rise -- the penalty reaches the
            # verbatim quotes a citation is made of, which is why repetition_penalty was rejected
            # outright.
            rows[-1]["call_failure_count"] = sum(
                1 for e in gen.errors if "rejected: " in e
            )
            rows[-1]["over_length_claims"] = sum(
                1 for e in gen.errors if "max claim length" in e
            )
            rows[-1]["runaway_chain_claims"] = sum(
                1 for e in gen.errors if "nested claims (non-terminating generation)" in e
            )
            rows[-1]["runaway_chain_pairs"] = sum(
                1 for e in gen.recovered if "extends " in e
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
            for note in gen.recovered:
                print(f"{'':>9s}   ~ {note}")

    per_system = {}
    for system in System:
        mine = [r for r in rows if r["system"] == system.value]
        lat = [r["latency_s"] for r in mine if r["latency_s"] is not None]
        per_system[system.value] = {
            "n": len(mine),
            "clean_parses": sum(1 for r in mine if not r["errors"]),
            "records_with_violations": sum(1 for r in mine if r["violations"]),
            "call_failure_count": sum(r["call_failure_count"] for r in mine),
            "stages_seen": sorted({r["stages"] for r in mine}),
            "mean_claims": round(statistics.fmean(r["claims"] for r in mine), 2),
            "total_citations": sum(r["citations"] for r in mine),
            "recovered_notes": sum(len(r["recovered"]) for r in mine),
            "over_length_claims": sum(r["over_length_claims"] for r in mine),
            "runaway_chain_claims": sum(r["runaway_chain_claims"] for r in mine),
            "runaway_chain_pairs": sum(r["runaway_chain_pairs"] for r in mine),
            "quote_not_found": sum(r["quote_not_found"] for r in mine),
            "longest_claim_words": max((r["longest_claim_words"] for r in mine), default=0),
            "median_latency_s": round(statistics.median(lat), 2) if lat else None,
        }

    print("\nper system:")
    for name, s in per_system.items():
        print(
            f"  {name:8s} clean_parses {s['clean_parses']}/{s['n']}  "
            f"violations {s['records_with_violations']}/{s['n']}  "
            f"call_failures {s['call_failure_count']}/{s['n']}  "
            f"stages {s['stages_seen']}  mean_claims {s['mean_claims']}  "
            f"runaway_chain_claims {s['runaway_chain_claims']}  "
            f"median {s['median_latency_s']}s"
        )

    expected = {System.JOINT.value: [1], System.POST_HOC.value: [2], System.VANILLA.value: [1]}
    stage_ok = all(per_system[k]["stages_seen"] == v for k, v in expected.items())
    total_call_failures = sum(s["call_failure_count"] for s in per_system.values())
    no_call_failures = total_call_failures == 0
    clean_pass = stage_ok and no_call_failures

    print(
        f"\nstage-count check: {'PASSES' if stage_ok else 'FAILS'} "
        "(post_hoc must be two calls, joint and vanilla one)"
    )
    print(
        f"call-failure check: {'PASSES' if no_call_failures else 'FAILS'} "
        f"({total_call_failures} model call(s) rejected)"
    )

    rec_path = records_path(prefix)
    cost_path = costs_path(prefix)
    sum_path = Path(str(prefix) + ".summary.json")
    write_jsonl(rec_path, records)
    write_jsonl(cost_path, costs)
    sum_path.write_text(
        json.dumps(
            {
                "script": "scripts/generate_smoke.py",
                "purpose": "W4 live-path smoke test; not a gate run and not a sample",
                "run_id": run_id,
                # The knobs are not restated here. The manifest holds the whole RunConfig and its
                # hash; a second copy beside it is a second thing to keep in sync, and the summary
                # used to carry knobs from three config sections flattened into one dict.
                "manifest": manifest_path(prefix).name,
                "config_hash": manifest["config_hash"],
                "fake_completer": args.fake,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "run_arguments": {
                    "base_url": None if args.fake else args.base_url,
                    "n_questions": len(contexts),
                    "contexts": str(args.contexts),
                    "max_claim_words": MAX_CLAIM_WORDS,
                },
                "stage_count_check": {"expected": expected, "passed": stage_ok},
                "call_failure_check": {"total_call_failures": total_call_failures, "passed": no_call_failures},
                "per_system": per_system,
                "rows": rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    finalize_run(prefix)
    print(f"\nWritten to {rec_path}, {cost_path}, {sum_path}, {manifest_path(prefix)}")
    for problem in verify_run(prefix):
        print(f"  provenance: {problem}")
    return 0 if clean_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
