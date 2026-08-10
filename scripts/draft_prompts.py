#!/usr/bin/env python3
"""Render the W3 generation prompts against real dev retrievals and size them.

This is the loop the equal-effort protocol runs on: render, read the output, revise, log a cycle in
`prompts.PROMPT_ITERATIONS`, render again. It calls no model — `generate.py` (W4) does that. What
it answers here is the part that does not need one: does the prompt render, how long is it against
the real 10-passage context, and does every stage still state the cap it is supposed to state.

Context source, in order of preference:

1. `docs/harvest/dev_contexts_top10.jsonl` — the real depth-10 context, produced on the box by
   `scripts/dump_contexts.py`.
2. `docs/harvest/table1_rows_1_4.records.jsonl` — carries text for ranks 1-5 only, because those
   records were written while G1 was a hit@5 gate. Usable for wording, **not** for sizing, and the
   artifact says so rather than quietly reporting a context 5 passages short of the real one.

Sizing is a **tokenizer measurement against the serving window**, not a chars/4 guess: the A4000
serves Llama-3.1-8B-AWQ at `--max-model-len 8192` (ADR-0008's runbook), and a prompt that overruns
it loses passages off the end of the context — a citation-recall loss the experiment would charge
to the system under test rather than to the harness.

CPU-only, runs on the laptop.

    uv run python scripts/draft_prompts.py
    uv run python scripts/draft_prompts.py --contexts docs/harvest/dev_contexts_top10.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from biomedqa.config import GenerationConfig  # noqa: E402
from biomedqa.data import load_instances, load_splits  # noqa: E402
from biomedqa.prompts import (  # noqa: E402
    CONTEXT_DEPTH,
    build_prompt,
    effort_is_matched,
    iteration_counts,
)
from biomedqa.schema import RetrievedPassage, System  # noqa: E402

#: A stand-in first-pass answer for post-hoc's cite stage, so the stage renders before any model
#: exists. Deliberately imperfect — one claim with no support in the passages — because the cite
#: prompt's hard instruction is "do not drop a claim", and a clean answer would not exercise it.
_STUB_ANSWER = (
    "DECISION: yes\n"
    "CLAIM 1: The intervention reduced the primary outcome relative to control.\n"
    "CLAIM 2: The effect was larger in participants over 65 years of age."
)

#: The generator G0 selected (ROADMAP W1) and the window it is served under
#: (`docs/harvest/runbooks/wsl-vllm-a4000.md`: `--max-model-len 8192`). A prompt that overruns this
#: is not a slow prompt, it is a **silently truncated context** — the passage the model was told to
#: cite disappears and the citation-recall loss is charged to the system under test.
GENERATOR = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
MAX_MODEL_LEN = 8192


def _token_counter(model: str):
    """`(count, how)` — the real tokenizer if it loads, else the chars/4 rule of thumb.

    The fallback is labelled in the artifact rather than silently substituted: biomedical prose
    tokenizes nearer 3.3 chars/token than 4, so the heuristic understates the budget by ~20% in
    exactly the direction that hides an overrun.
    """
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model)
    except Exception as exc:  # offline box, or no transformers — say so, do not guess quietly
        print(f"Tokenizer {model} unavailable ({type(exc).__name__}); using chars/4.", file=sys.stderr)
        return (lambda s: len(s) // 4), "chars/4 estimate"
    return (lambda s: len(tok(s, add_special_tokens=False)["input_ids"])), model


def _from_contexts(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _from_table1(path: Path, row: int, split: str) -> list[dict]:
    """Fallback: ranks 1-5 only. Depth is reported honestly by the caller."""
    pubids = set(load_splits()[split])
    questions = {i.pubid: i.question for i in load_instances() if i.pubid in pubids}
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("table1_row") != row:
                continue
            ps = [p for p in sorted(rec["retrieved"], key=lambda p: p["rank"]) if p.get("text")]
            out.append(
                {
                    "query_id": rec["query_id"],
                    "question": questions.get(rec["query_id"], ""),
                    "depth": len(ps),
                    "passages": ps,
                }
            )
    return out


def _passages(row: dict) -> list[RetrievedPassage]:
    return [
        RetrievedPassage(
            passage_id=p["passage_id"],
            rank=p["rank"],
            score=p.get("score", 0.0),
            retriever=p.get("retriever", "rerank"),
            text=p["text"],
        )
        for p in row["passages"]
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description="Render W3 generation prompts on dev retrievals")
    ap.add_argument("--contexts", type=Path, default=Path("docs/harvest/dev_contexts_top10.jsonl"))
    ap.add_argument(
        "--records", type=Path, default=Path("docs/harvest/table1_rows_1_4.records.jsonl")
    )
    ap.add_argument("--row", type=int, default=4)
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--samples", type=int, default=3, help="Queries written out in full")
    ap.add_argument("--tokenizer", default=GENERATOR, help="HF id used to count prompt tokens")
    ap.add_argument("--out", type=Path, default=Path("docs/harvest/prompt_drafts.json"))
    args = ap.parse_args()

    if args.contexts.exists():
        rows, source, full_depth = _from_contexts(args.contexts), str(args.contexts), True
    else:
        rows = _from_table1(args.records, args.row, args.split)
        source, full_depth = str(args.records), False
        print(
            f"{args.contexts} not found — falling back to {args.records}, which carries text for "
            f"ranks 1-5 only. Wording is exercisable; sizing is not.\n"
            f"On the A4000: uv run python scripts/dump_contexts.py --index-dir data/index/empty "
            f"--depth {CONTEXT_DEPTH}\n",
            file=sys.stderr,
        )
    if not rows:
        print("No dev contexts found.", file=sys.stderr)
        return 1

    cfg = GenerationConfig()
    count_tokens, counted_by = _token_counter(args.tokenizer)
    stages = [
        ("joint", System.JOINT, "answer", None),
        ("post_hoc_answer", System.POST_HOC, "answer", None),
        ("post_hoc_cite", System.POST_HOC, "cite", _STUB_ANSWER),
        ("vanilla", System.VANILLA, "answer", None),
    ]

    sizes: dict[str, list[int]] = {name: [] for name, *_ in stages}
    tokens: dict[str, list[int]] = {name: [] for name, *_ in stages}
    for row in rows:
        ps = _passages(row)
        for name, system, stage, answer in stages:
            prompt = build_prompt(
                system, row["question"], ps, cfg.max_citations, stage=stage, answer=answer
            )
            sizes[name].append(len(prompt))
            tokens[name].append(count_tokens(prompt))

    depths = sorted({r["depth"] for r in rows})
    budget = MAX_MODEL_LEN - cfg.max_tokens
    print(f"{len(rows)} dev queries, context depth {depths}")
    print(f"prompt tokens by {counted_by}; budget {MAX_MODEL_LEN} - {cfg.max_tokens} completion = {budget}")
    print(f"{'stage':<18}{'chars max':>11}{'tok mean':>10}{'tok max':>9}{'headroom':>10}{'fits':>7}")
    print("-" * 65)
    overruns = {}
    for name in sizes:
        t = tokens[name]
        worst = max(t)
        over = sum(1 for v in t if v > budget)
        overruns[name] = over
        print(
            f"{name:<18}{max(sizes[name]):>11,}{sum(t) // len(t):>10,}{worst:>9,}"
            f"{budget - worst:>10,}{('yes' if over == 0 else f'NO ({over})'):>7}"
        )

    counts = iteration_counts()
    print(f"\nprompt-iteration cycles: {counts}  matched(joint, post_hoc)={effort_is_matched()}")

    samples = []
    for row in rows[: args.samples]:
        ps = _passages(row)
        samples.append(
            {
                "query_id": row["query_id"],
                "question": row["question"],
                "depth": row["depth"],
                "prompts": {
                    name: build_prompt(
                        system, row["question"], ps, cfg.max_citations, stage=stage, answer=answer
                    )
                    for name, system, stage, answer in stages
                },
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "script": "scripts/draft_prompts.py",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "context_source": source,
                "context_depth_is_the_real_one": full_depth,
                "context_depths_seen": depths,
                "target_depth": CONTEXT_DEPTH,
                "max_citations": cfg.max_citations,
                "prompt_iteration_cycles": counts,
                "effort_matched_joint_vs_post_hoc": effort_is_matched(),
                "prompt_chars": {
                    k: {"min": min(v), "mean": sum(v) // len(v), "max": max(v)}
                    for k, v in sizes.items()
                },
                "token_budget": {
                    "counted_by": counted_by,
                    "max_model_len": MAX_MODEL_LEN,
                    "completion_reserved": cfg.max_tokens,
                    "prompt_budget": budget,
                    "queries_over_budget": overruns,
                    "fits": not any(overruns.values()),
                },
                "prompt_tokens": {
                    k: {"min": min(v), "mean": sum(v) // len(v), "max": max(v)}
                    for k, v in tokens.items()
                },
                "samples": samples,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nWritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
