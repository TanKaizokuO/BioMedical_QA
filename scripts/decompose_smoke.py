#!/usr/bin/env python3
"""C7 smoke test: drive `decompose()` + `generate.cite_claims()` through the live vLLM server on
the A4000, once per granularity row, over post-hoc's already-generated `parity_iter1b` answers.

Mirrors `generate_smoke.py`'s reasoning exactly (read that docstring first): every component here
has only ever seen an injected completer, and the question this answers has never been asked of real
hardware — does an 8B model honour `CLAIM <n> FROM <k>`, does the `atomic` row actually stay bare, and
does the re-citation call (`generate.cite_claims`, Option A per `HANDOFF.md`) survive contact with
the model twice per query for the two model-driven rows?

Three things it measures that a stub cannot:

1. **Parse rate**, per row — the share of `decompose()` calls that returned zero errors. Feeds G2's
   ≥95% valid-parse bar the same way `generate_smoke.py`'s `clean_parses` does for the headline path.
2. **`atomic` divergence** — the share of `atomic` claims whose text is not a substring of the
   sentence it came from (case-sensitive, after stripping whitespace). A bare-atomic claim that only
   *splits* its source sentence is always a substring of it; one that resolves a pronoun or supplies
   an implicit subject is not. This is a proxy, not a ground truth — see ADR-0018 §2 — but a `atomic`
   row whose divergence rate approaches `decontextualized_atomic`'s is the ablation failing its own
   validity condition, not a subtler success.
3. **Median words/claim per row**, via `scoring.granularity.words_in_claim` — checked against the
   inequality `sentence ≥ atomic ≥ decontextualized_atomic` the three row names claim to produce.

**`--fake`'s numbers are not that measurement.** The canned completers echo each source sentence
back as one claim per sentence and never decontextualize (there is no model behind them to do it),
so under `--fake` the three rows read as textually near-identical and the monotonicity check is not
informative — it exercises the orchestration and the file shapes, nothing about model behaviour.
Only a live run answers 1–3; `--fake` is what makes an orchestration bug discoverable before a GPU
run rather than during one.

**Input is post-hoc's *cited* answer** (`generate.split_stages(raw_generation)[-1]`), not the answer
stage alone: `decompose.answer_spans` reads `CLAIM` line bodies and ignores `CITE:`/`DECISION:`
lines when the grammar is present (`decompose.py`'s module docstring, decision 2), so the cite
stage's extra lines cost nothing and its claim text is post-hoc's final, already-cited output.

Each (query, row) becomes one `QueryRecord`, `query_id` suffixed `:<granularity>` so the three rows
for one question do not collide under `system=post_hoc` in `records.jsonl` — strip the suffix to
recover the source query. `system` stays `post_hoc`: these rows are a re-cut of post-hoc's answer,
never a different generation.
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
from biomedqa.decompose import decompose  # noqa: E402
from biomedqa.generate import cite_claims, split_stages  # noqa: E402
from biomedqa.harness import (  # noqa: E402
    costs_path,
    finalize_run,
    manifest_path,
    records_path,
    run_manifest,
    verify_run,
)
from biomedqa.schema import (  # noqa: E402
    CostRecord,
    Granularity,
    QueryRecord,
    System,
    read_query_records,
    write_jsonl,
)
from biomedqa.scoring.granularity import words_in_claim  # noqa: E402

#: The two rows that call a model. `sentence` makes no call (`decompose.unit_rules`'s own reasoning
#: — it is the control for decomposition error), so it never needs a completer.
MODEL_ROWS = (Granularity.ATOMIC, Granularity.DECONTEXTUALIZED_ATOMIC)
ALL_ROWS = (Granularity.SENTENCE, *MODEL_ROWS)


def load_post_hoc(path: Path, n: int) -> list[QueryRecord]:
    """The first `n` `post_hoc` records from `parity_iter1b`, in file order.

    File order, not a sample — same reasoning as `generate_smoke.load_contexts`: this is a smoke
    test, and `--n 5` does not generalise to the dev set.
    """
    out = []
    for rec in read_query_records(path):
        if rec.system is System.POST_HOC:
            out.append(rec)
        if len(out) == n:
            break
    if len(out) < n:
        raise SystemExit(f"{path} holds {len(out)} post_hoc records, asked for {n}")
    return out


def assert_served(base_url: str, model: str, timeout: float) -> None:
    try:
        with httpx.Client(base_url=base_url, timeout=timeout) as client:
            served = [m["id"] for m in client.get("/v1/models").json()["data"]]
    except Exception as exc:
        raise SystemExit(
            f"cannot reach a vLLM server at {base_url}: {exc}\n"
            "Start it per docs/harvest/runbooks/wsl-vllm-a4000.md §5, or tunnel to the box."
        ) from exc
    if model not in served:
        raise SystemExit(f"{base_url} does not serve {model!r}.\nServed: {served}")
    print(f"vLLM at {base_url} serving {model}")


def _fake_decompose_completer(prompt: str, config: GenerationConfig, **kw) -> tuple[str, CostRecord]:
    """One claim per numbered sentence, verbatim. Exercises `CLAIM <n> FROM <k>` orchestration and
    parsing — never decontextualizes, because there is no model behind it to do so (module
    docstring, "`--fake`'s numbers are not that measurement")."""
    m = re.search(r"Answer, one sentence per line:\n(.*?)\n\nSplit the answer", prompt, re.S)
    if m is None:
        raise AssertionError("canned decompose completer could not find the numbered sentence list")
    lines = []
    for line in m.group(1).splitlines():
        idx, sep, text = line.partition(". ")
        if sep and idx.strip().isdigit() and text.strip():
            lines.append(f"CLAIM {idx.strip()} FROM {idx.strip()}: {text.strip()}")
    return (
        "\n".join(lines),
        CostRecord(
            run_id=kw.get("run_id", ""), query_id=kw.get("query_id"), component="generate",
            backend=f"vllm:{config.model}", input_tokens=512, output_tokens=32, wall_s=0.0,
        ),
    )


def _fake_cite_completer(prompt: str, config: GenerationConfig, **kw) -> tuple[str, CostRecord]:
    """Reproduces the given CLAIM lines and cites the first context passage verbatim in each."""
    ctx_m = re.search(r"^\[([^\]\s]+)\]\n(.+)$", prompt, re.M)
    if ctx_m is None:
        raise AssertionError("canned cite completer found no bracketed passage id in the prompt")
    pid, text = ctx_m.group(1), ctx_m.group(2)
    ans_m = re.search(r"Answer already given:\n(.*?)\n\nFor each claim", prompt, re.S)
    if ans_m is None:
        raise AssertionError("canned cite completer could not find the reproduced answer")
    lines = ["DECISION: yes"]
    for claim_line in ans_m.group(1).splitlines():
        if claim_line.strip():
            lines.append(claim_line.strip())
            lines.append(f"CITE: [{pid}] || {text[:60].strip()}")
    return (
        "\n".join(lines),
        CostRecord(
            run_id=kw.get("run_id", ""), query_id=kw.get("query_id"), component="generate",
            backend=f"vllm:{config.model}", input_tokens=768, output_tokens=48, wall_s=0.0,
        ),
    )


def _divergence(claims, answer: str) -> tuple[int, int]:
    """`(diverged, checkable)` for `atomic` — a claim diverges when its text is not a substring of
    the source sentence it points at (ADR-0018 §2's proxy). Claims with no span are excluded from
    the denominator: a `decompose` parse failure already reported separately, not a divergence."""
    diverged = checkable = 0
    for c in claims:
        if c.source_start is None or c.source_end is None:
            continue
        checkable += 1
        source = answer[c.source_start:c.source_end].strip()
        if c.text.strip() not in source:
            diverged += 1
    return diverged, checkable


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="model id exactly as served by /v1/models")
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--contexts", type=Path, default=Path("docs/harvest/parity_iter1b.records.jsonl"))
    ap.add_argument("--n", type=int, default=100, help="post_hoc questions (default: %(default)s — the full parity_iter1b dev slice)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=GenerationConfig().max_tokens)
    ap.add_argument(
        "--frequency-penalty", type=float, default=GenerationConfig().frequency_penalty,
        help="OpenAI-standard frequency_penalty, forwarded to vLLM as-is (never repetition_penalty "
             "— see GenerationConfig.frequency_penalty). Sweep this to escape the greedy-decoding "
             "repetition loop documented in docs/harvest/decompose_smoke.summary.json.",
    )
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--out-prefix", type=Path, default=Path("docs/harvest/decompose_smoke"))
    ap.add_argument("--fake", action="store_true", help="canned completers; no network, no GPU")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    decompose_completer = _fake_decompose_completer if args.fake else None
    cite_completer = _fake_cite_completer if args.fake else None
    prefix = args.out_prefix
    if args.fake and not prefix.name.endswith("_fakecompleter"):
        prefix = prefix.with_name(prefix.name + "_fakecompleter")
    run_id = prefix.name

    if args.fake:
        print("--fake: canned completers, no network. Numbers are orchestration proof, not a measurement.")
    else:
        os.environ["VLLM_BASE_URL"] = args.base_url
        assert_served(args.base_url, args.model, args.timeout)

    source_records = load_post_hoc(args.contexts, args.n)

    run_config = RunConfig().ablate(
        run_id,
        **{
            "generation.backend": "vllm",
            "generation.model": args.model,
            "generation.max_tokens": args.max_tokens,
            "generation.frequency_penalty": args.frequency_penalty,
            "generation.seeds": (args.seed,),
            # Nominal — the three rows share this manifest under one `run_id`, told apart by the
            # `:<granularity>` query_id suffix, same pattern `generate_smoke.py` uses for `system`.
            "generation.granularity": Granularity.DECONTEXTUALIZED_ATOMIC.value,
        },
    )

    if args.overwrite:
        for path in (manifest_path(prefix), records_path(prefix), costs_path(prefix)):
            path.unlink(missing_ok=True)
    try:
        manifest = run_manifest(run_config, prefix)
    except FileExistsError as exc:
        raise SystemExit(f"{exc} Pass --overwrite to do exactly that.")
    print(f"manifest: {manifest_path(prefix)}  config {manifest['config_hash']}  git {manifest['git_sha']}")

    records: list[QueryRecord] = []
    costs: list[CostRecord] = []
    per_row: dict[str, dict] = {
        row.value: {
            "claims": [], "clean_decompose": 0, "clean_cite": 0, "n": 0,
            "duplicate_claim_count": 0, "quote_not_found_count": 0,
        }
        for row in ALL_ROWS
    }

    row_configs = {
        row: run_config.ablate(f"{run_id}-{row.value}", **{"generation.granularity": row.value}).generation
        for row in ALL_ROWS
    }

    for src in source_records:
        answer = split_stages(src.raw_generation)[-1]
        for row in ALL_ROWS:
            row_config = row_configs[row]
            row_id = f"{src.query_id}:{row.value}"
            dkw = {"completer": decompose_completer} if decompose_completer else {}
            decomp = decompose(
                answer, row_config, question=src.question, seed=args.seed,
                run_id=run_id, query_id=row_id, **dkw,
            )
            costs.extend(decomp.costs)
            per_row[row.value]["n"] += 1
            if not decomp.errors:
                per_row[row.value]["clean_decompose"] += 1
            per_row[row.value]["duplicate_claim_count"] += sum(
                1 for problem in decomp.errors if "claim text verbatim" in problem
            )

            claims = list(decomp.claims)
            if row in MODEL_ROWS and claims and src.retrieved:
                ckw = {"complete": cite_completer} if cite_completer else {}
                recite = cite_claims(
                    claims, src.question, src.retrieved, row_config, seed=args.seed,
                    run_id=run_id, query_id=row_id, **ckw,
                )
                costs.append(recite.cost)
                if not recite.errors:
                    per_row[row.value]["clean_cite"] += 1
                per_row[row.value]["quote_not_found_count"] += sum(
                    1 for problem in recite.errors if "quote not found verbatim" in problem
                )
                claims = list(recite.claims)
            elif row is Granularity.SENTENCE:
                per_row[row.value]["clean_cite"] += 1  # no cite call attempted for the control row

            per_row[row.value]["claims"].extend(claims)
            records.append(
                QueryRecord(
                    run_id=run_id, query_id=row_id, question=src.question, system=System.POST_HOC,
                    seed=args.seed, retrieved=src.retrieved, gold_passage_ids=src.gold_passage_ids,
                    claims=claims, raw_generation=src.raw_generation,
                    final_decision=src.final_decision, gold_final_decision=src.gold_final_decision,
                )
            )
            print(
                f"{src.query_id:>9s} {row.value:24s} claims={len(claims):2d} "
                f"decompose_errors={len(decomp.errors)}"
            )
            for problem in decomp.errors:
                print(f"{'':>9s}   ! {problem}")

    summary = {}
    for row in ALL_ROWS:
        stats = per_row[row.value]
        words = [words_in_claim(c.text) for c in stats["claims"]]
        entry = {
            "n_queries": stats["n"],
            "total_claims": len(stats["claims"]),
            "clean_decompose_rate": round(stats["clean_decompose"] / stats["n"], 4) if stats["n"] else None,
            "clean_cite_rate": round(stats["clean_cite"] / stats["n"], 4) if stats["n"] else None,
            "median_words_per_claim": statistics.median(words) if words else None,
            "duplicate_claim_count": stats["duplicate_claim_count"],
            "quote_not_found_count": stats["quote_not_found_count"],
        }
        if row is Granularity.ATOMIC:
            diverged = checkable = 0
            # Recomputed from the stored records against each source's cited-stage answer text.
            by_query = {}
            for rec in records:
                if rec.query_id.endswith(":atomic"):
                    by_query[rec.query_id[: -len(":atomic")]] = rec
            for src in source_records:
                atomic_rec = by_query.get(src.query_id)
                if atomic_rec is None:
                    continue
                d, c = _divergence(atomic_rec.claims, split_stages(src.raw_generation)[-1])
                diverged += d
                checkable += c
            entry["divergence_rate"] = round(diverged / checkable, 4) if checkable else None
        summary[row.value] = entry
        print(f"\n{row.value}: {entry}")

    medians = {row.value: summary[row.value]["median_words_per_claim"] for row in ALL_ROWS}
    monotonic = (
        medians[Granularity.SENTENCE.value] is not None
        and medians[Granularity.SENTENCE.value] >= medians[Granularity.ATOMIC.value] >= medians[Granularity.DECONTEXTUALIZED_ATOMIC.value]
    )
    print(
        f"\nmonotonicity check (sentence >= atomic >= decontextualized_atomic): "
        f"{'PASSES' if monotonic else 'FAILS'} — {medians}"
        + ("" if not args.fake else " (uninformative under --fake, see module docstring)")
    )

    rec_path, cost_path = records_path(prefix), costs_path(prefix)
    sum_path = Path(str(prefix) + ".summary.json")
    write_jsonl(rec_path, records)
    write_jsonl(cost_path, costs)
    sum_path.write_text(
        json.dumps(
            {
                "script": "scripts/decompose_smoke.py",
                "purpose": "C7 live-path smoke test; not a gate run and not a sample",
                "run_id": run_id,
                "manifest": manifest_path(prefix).name,
                "config_hash": manifest["config_hash"],
                "fake_completer": args.fake,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "run_arguments": {
                    "base_url": None if args.fake else args.base_url,
                    "n_questions": args.n,
                    "contexts": str(args.contexts),
                    "max_tokens": args.max_tokens,
                    "frequency_penalty": args.frequency_penalty,
                },
                "per_row": summary,
                "monotonicity_check": {"passes": monotonic, "medians": medians},
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
    return 0 if monotonic or args.fake else 1


if __name__ == "__main__":
    raise SystemExit(main())
